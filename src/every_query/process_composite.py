from pathlib import Path
from typing import Any
import hashlib
import re

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score


def values_as_list(**kwargs) -> list[Any]:
    return list(kwargs.values())


def code_slug(code: str, n_hash: int = 10, prefix_len: int = 24) -> str:
    h = hashlib.sha1(code.encode("utf-8")).hexdigest()[:n_hash]
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", code).strip("_")[:prefix_len]
    return f"{prefix}__{h}" if prefix else h



@hydra.main(version_base="1.3", config_path=".", config_name="process_composite_config.yaml")
def main(cfg: DictConfig) -> None:
    all_preds_df=pl.read_csv(cfg.predictions_df_path)
    
    probs_df = (
        all_preds_df
        .group_by(["subject_id","prediction_time"])
        .agg(
            pl.col("occurs_probs").max().alias("max_prob")
        )
    )
    
    all_aces_labels_df=pl.read_parquet(cfg.task_labels_df_path,columns=["subject_id","prediction_time","boolean_value"])

    probs_df = probs_df.with_columns(
    pl.col("prediction_time")
      .cast(pl.Datetime("us"))
      .alias("prediction_time")
      )

    joined_df = probs_df.join(all_aces_labels_df,on=["subject_id","prediction_time"],how="inner")

    
    auc = roc_auc_score(
        joined_df["boolean_value"].to_numpy(),
        joined_df["max_prob"].to_numpy(),
    )

    print(auc)







    
if __name__ == "__main__":
    main()
