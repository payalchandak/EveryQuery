import logging
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_eval_results(model_run_dirs: list[Path], split: str) -> pl.DataFrame:
    dfs: list[pl.DataFrame] = []
    for run_dir in model_run_dirs:
        if not run_dir.is_dir():
            logger.warning(f"{run_dir} is not a directory, skipping")
            continue
        matches = sorted(run_dir.glob(f"eval/best/eval_aucs_{split}_*.parquet"))
        if not matches:
            logger.warning(f"No eval/best/eval_aucs_{split}_*.parquet under {run_dir}")
            continue
        path = matches[-1]
        logger.info(f"Loading {path}")
        dfs.append(pl.read_parquet(path))
    if not dfs:
        raise FileNotFoundError(f"No eval parquets found for split={split}")
    return pl.concat(dfs, how="vertical")


def _compute_win_rates(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    pivot = df.pivot(values="occurs_auc", index=["code", "duration_days"], on="model")

    before = pivot.height
    pivot = pivot.drop_nulls()
    dropped = before - pivot.height
    if dropped:
        logger.warning(f"Dropped {dropped}/{before} (code, duration) pairs not covered by every model")

    models = [c for c in pivot.columns if c not in ("code", "duration_days")]
    n_pairs = pivot.height
    if n_pairs == 0:
        raise ValueError("No (code, duration) pairs with coverage across all models")
    if len(models) < 2:
        raise ValueError(f"Need at least 2 models to compare, got {len(models)}")

    fractions: dict[str, dict[str, float]] = {m: dict.fromkeys(models, 0.0) for m in models}
    for a, b in combinations(models, 2):
        col_a = pivot[a]
        col_b = pivot[b]
        wins_a = (col_a > col_b).sum() + 0.5 * (col_a == col_b).sum()
        frac_a = wins_a / n_pairs
        fractions[a][b] = frac_a
        fractions[b][a] = 1.0 - frac_a

    win_matrix = pl.DataFrame({"model": models, **{m: [fractions[row][m] for row in models] for m in models}})

    total_wins = [sum(fractions[m].values()) for m in models]
    win_rate_vals = [w / (len(models) - 1) for w in total_wins]
    ranking = (
        pl.DataFrame({"model": models, "win_rate": win_rate_vals})
        .sort("win_rate", descending=True)
        .with_row_index(name="rank", offset=1)
        .select("model", "win_rate", "rank")
    )
    return ranking, win_matrix


@hydra.main(version_base="1.3", config_path="./conf", config_name="select_model_config.yaml")
def main(cfg: DictConfig) -> None:
    model_run_dirs = [Path(p) for p in cfg.model_run_dirs]
    split = cfg.split

    df = _load_eval_results(model_run_dirs, split)
    logger.info(
        f"Loaded {df.height} rows across {df['model'].n_unique()} models, "
        f"{df['code'].n_unique()} codes, {df['duration_days'].n_unique()} durations"
    )

    ranking, win_matrix = _compute_win_rates(df)

    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(f"\n=== Overall win rate ranking ({split}, aggregated across durations) ===")
        print(ranking.with_columns(pl.col("win_rate").round(3)))
        print("\nPairwise win matrix (row beats column):")
        print(win_matrix.with_columns([pl.col(c).round(3) for c in win_matrix.columns if c != "model"]))

    best = ranking.row(0, named=True)
    print(f"\n>>> Best model: {best['model']} (win rate: {best['win_rate']:.3f})")

    out_dir = Path(cfg.output_dir) if cfg.get("output_dir") else Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    ranking_fp = out_dir / f"model_selection_{split}_{ts}.parquet"
    matrix_fp = out_dir / f"win_matrix_{split}_{ts}.parquet"
    ranking.write_parquet(ranking_fp)
    win_matrix.write_parquet(matrix_fp)
    logger.info(f"Saved ranking to {ranking_fp}")
    logger.info(f"Saved win matrix to {matrix_fp}")


if __name__ == "__main__":
    main()
