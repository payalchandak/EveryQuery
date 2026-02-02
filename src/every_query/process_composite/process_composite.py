import hashlib
import re
from typing import Any

import hydra
import polars as pl
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score


def values_as_list(**kwargs) -> list[Any]:
    return list(kwargs.values())


def code_slug(code: str, n_hash: int = 10, prefix_len: int = 24) -> str:
    h = hashlib.sha1(code.encode("utf-8")).hexdigest()[:n_hash]
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", code).strip("_")[:prefix_len]
    return f"{prefix}__{h}" if prefix else h


def agg_probs(
    all_preds_df_fp: str,
    all_aces_labels_df_fp: str,
    agg_type: list[str],
    *,
    pred_prob_col: str = "occurs_probs",
    label_col: str = "boolean_value",
    clip_sum_to_1: bool = False,
    return_auc: bool = True,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, float]]:
    """Aggregate per-(subject_id, prediction_time) probabilities across multiple queries/codes.

    agg_type supports:
      - "max": max(p_i)
      - "or":  1 - prod(1 - p_i)  (conditional independence)
      - "sum": sum(p_i)

    Returns:
      - df with columns: subject_id, prediction_time, prob_max/prob_or/prob_sum
      - optionally also returns dict of AUROCs per agg column.
    """
    agg_type_set = {a.lower() for a in agg_type}
    allowed = {"max", "or", "sum"}
    bad = sorted(agg_type_set - allowed)
    if bad:
        raise ValueError(f"Unknown agg_type(s): {bad}. Allowed: {sorted(allowed)}")

    preds = pl.read_csv(all_preds_df_fp)
    labels = pl.read_parquet(all_aces_labels_df_fp, columns=["subject_id", "prediction_time", label_col])

    aggs: list[pl.Expr] = []

    if "max" in agg_type_set:
        aggs.append(pl.col(pred_prob_col).max().alias("prob_max"))

    if "or" in agg_type_set:
        # 1 - prod(1 - p). Clip inside to avoid negatives from float error.
        aggs.append((1.0 - (1.0 - pl.col(pred_prob_col)).product()).alias("prob_or"))

    if "sum" in agg_type_set:
        expr = pl.col(pred_prob_col).sum()

        aggs.append(expr.alias("prob_sum"))

    probs_df = (
        preds.group_by(["subject_id", "prediction_time"]).agg(aggs).sort(["subject_id", "prediction_time"])
    )

    joined_df = probs_df.join(labels, on=["subject_id", "prediction_time"], how="inner")

    if not return_auc:
        return joined_df

    aucs: dict[str, float] = {}
    y = joined_df[label_col].to_numpy()
    for col in [c for c in joined_df.columns if c.startswith("prob_")]:
        p = joined_df[col].to_numpy()
        # roc_auc_score errors if y has only one class in the joined set
        try:
            aucs[col] = float(roc_auc_score(y, p))
        except ValueError:
            aucs[col] = float("nan")

    return joined_df, aucs


@hydra.main(version_base="1.3", config_path=".", config_name="process_composite_config.yaml")
def main(cfg: DictConfig) -> None:
    _, aucs = agg_probs(
        all_preds_df_fp=cfg.predictions_df_path,
        all_aces_labels_df_fp=cfg.task_labels_df_path,
        agg_type=["max", "or", "sum"],
    )
    print(aucs)


if __name__ == "__main__":
    main()
