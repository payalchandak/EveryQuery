"""``EQ_build_ontology`` — derive an ontology DAG from a cohort's code metadata.

Reads ``{tensorized_cohort_dir}/metadata/codes.parquet`` and writes the three artifacts
:mod:`every_query.data.ontology` documents into ``out_dir``.  Run once per cohort (and once per
``decay`` you want to sweep); training and generation then point at that directory.

    EQ_build_ontology \\
        tensorized_cohort_dir=$TENSORIZED_COHORT_DIR \\
        out_dir=$ONTOLOGY_DIR \\
        decay=0.5

The artifacts are a **side-car**: the cohort's own ``codes.parquet`` is read, never written, so
adding or rebuilding an ontology cannot disturb runs that do not use one.
"""

import logging
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from every_query.data.ontology import (
    EMBEDDING_MIX_FILE,
    EVENT_TO_QUERY_NODES_FILE,
    ONTOLOGY_VOCAB_FILE,
    build_event_to_query_nodes,
    build_ontology,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIGS = str(files("every_query") / "data" / "configs")


def run(
    tensorized_cohort_dir: Path,
    out_dir: Path,
    decay: float = 0.5,
    subtree_suffix: str | None = "ANY",
) -> Path:
    """Build and write the ontology artifacts; returns ``out_dir``."""
    codes_fp = Path(tensorized_cohort_dir) / "metadata" / "codes.parquet"
    if not codes_fp.is_file():
        raise FileNotFoundError(
            f"No code metadata at {codes_fp}.  `tensorized_cohort_dir` must point at a "
            f"tensorized cohort root (the same one training reads)."
        )

    columns = pl.read_parquet_schema(codes_fp).keys()
    wanted = ["code", "code/vocab_index"] + (["parent_codes"] if "parent_codes" in columns else [])
    if "parent_codes" not in columns:
        logger.info(
            "No `parent_codes` column in %s; the DAG will come from //-prefix structure alone.",
            codes_fp,
        )
    codes_df = pl.read_parquet(codes_fp, columns=wanted)

    nodes_df, mix_df = build_ontology(codes_df, decay=decay, subtree_suffix=subtree_suffix)
    closure_df = build_event_to_query_nodes(nodes_df, mix_df)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_df.write_parquet(out_dir / ONTOLOGY_VOCAB_FILE)
    mix_df.write_parquet(out_dir / EMBEDDING_MIX_FILE)
    closure_df.write_parquet(out_dir / EVENT_TO_QUERY_NODES_FILE)

    n_leaves = int(nodes_df["is_observed_code"].sum())
    n_subtree = (
        0
        if not subtree_suffix
        else int(nodes_df.filter(pl.col("node_name").str.ends_with(f"//{subtree_suffix}")).height)
    )
    logger.info(
        "Wrote ontology to %s: %d nodes (%d leaves + %d ancestors, of which %d are dual-role "
        "subtree nodes), %d mix entries, %d closure rows, decay=%g, subtree_suffix=%r.",
        out_dir,
        nodes_df.height,
        n_leaves,
        nodes_df.height - n_leaves,
        n_subtree,
        mix_df.height,
        closure_df.height,
        decay,
        subtree_suffix,
    )
    return out_dir


@hydra.main(version_base="1.3", config_path=CONFIGS, config_name="build_ontology")
def main(cfg: DictConfig) -> None:
    if not cfg.get("tensorized_cohort_dir"):
        raise ValueError("`tensorized_cohort_dir=` is required (pass $TENSORIZED_COHORT_DIR).")
    if not cfg.get("out_dir"):
        raise ValueError("`out_dir=` is required — the directory to write the artifacts into.")
    suffix = cfg.get("subtree_suffix", "ANY")
    run(
        tensorized_cohort_dir=Path(cfg.tensorized_cohort_dir),
        out_dir=Path(cfg.out_dir),
        decay=float(cfg.get("decay", 0.5)),
        subtree_suffix=None if suffix in (None, "", "null") else str(suffix),
    )


if __name__ == "__main__":
    main()
