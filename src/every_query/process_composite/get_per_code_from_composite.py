import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import hydra
import polars as pl
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def values_as_list(**kwargs) -> list[Any]:
    return list(kwargs.values())


def code_slug(code: str, n_hash: int = 10, prefix_len: int = 24) -> str:
    h = hashlib.sha1(code.encode("utf-8")).hexdigest()[:n_hash]
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", code).strip("_")[:prefix_len]
    return f"{prefix}__{h}" if prefix else h


@hydra.main(
    version_base="1.3", config_path="./eval_suite/conf", config_name="get_per_code_from_composite_config.yaml"
)
def main(cfg: DictConfig) -> None:
    print("starting")
    task_code_dfs_parent = Path(cfg.task_df_fp)
    code_task_df_paths = sorted(task_code_dfs_parent.glob("*"))

    all_preds_df = pl.read_csv(cfg.preds_df_fp)
    all_preds_df = all_preds_df.with_columns(pl.col("code").map_elements(code_slug).alias("code_slugged"))

    all_preds_df = all_preds_df.with_columns(
        pl.from_epoch(pl.col("prediction_time"), time_unit="us").alias("prediction_time")
    )

    results = {}

    for code_task_path in tqdm(code_task_df_paths, total=len(code_task_df_paths)):
        slugged_code = Path(code_task_path).stem
        code_task_df = pl.read_parquet(code_task_path, columns=["subject_id", "prediction_time", "occurs"])

        code_preds_df = all_preds_df.filter(pl.col("code_slugged") == slugged_code)
        # print(code_preds_df.head())
        # print(slugged_code)

        joined_df = code_preds_df.join(code_task_df, on=["subject_id", "prediction_time"], how="left")
        print(joined_df.head())
        print(joined_df.shape)
        code = joined_df["code"][0]
        print(code)

        y_true = joined_df["occurs"].to_numpy()
        y_score = joined_df["occurs_probs"].to_numpy()

        auc = roc_auc_score(y_true, y_score)
        results[code] = auc

    out_fp = cfg.output_root + "readmiss_30d_per_code.csv"
    out_df = pl.DataFrame(results)
    out_df.write_csv(out_fp)


if __name__ == "__main__":
    main()
