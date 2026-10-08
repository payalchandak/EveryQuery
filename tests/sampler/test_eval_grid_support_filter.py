"""Global conditional-task filtering before multitask inference."""

import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
import yaml

from every_query.data.schema import QuerySeqSchema
from every_query.generate_tasks import sample_evaluation_query_sequences as eval_seq
from tests.designed_specs import entry
from tests.sampler.test_eval_grid_per_shard import SPLIT, _labels, _run_seq, _unique


@pytest.fixture
def data_dir(tmp_path: Path, synthetic_events: pl.DataFrame, write_split_shards) -> Path:
    return write_split_shards(
        tmp_path,
        {
            "0": synthetic_events.filter(pl.col("subject_id") != 3),
            "1": synthetic_events.filter(pl.col("subject_id") == 3),
        },
        split=SPLIT,
    )


@pytest.fixture
def codes_yaml(tmp_path: Path, synthetic_query_codes: list[str]) -> Path:
    fp = tmp_path / "codes.yaml"
    fp.write_text(yaml.safe_dump(synthetic_query_codes))
    return fp


def test_global_support_counts_conditional_tasks_and_keeps_negatives(tmp_path: Path):
    """The two positives live on different shards; conditioning and forcing split tasks."""
    files = [tmp_path / f"{i}.parquet" for i in range(2)]
    rows = [
        [([False, True], "A"), ([False, False], "A"), ([True, True], "A"), ([False, True], "B")],
        [([False, True], "A"), ([False, False], "A"), ([False, True], "A_forced")],
    ]
    for fp, shard_rows in zip(files, rows, strict=True):
        pl.DataFrame(
            {
                "queries": [["condition", code.split("_")[0]] for _, code in shard_rows],
                "durations": [[1.0, 30.0]] * len(shard_rows),
                "answers": [answers for answers, _ in shard_rows],
                "forced_answers": [
                    [False, None] if code == "A_forced" else [None, None] for _, code in shard_rows
                ],
            }
        ).write_parquet(fp)

    summary = eval_seq.filter_tasks_by_min_positives(files, 2)
    assert summary == {"tasks_before": 4, "tasks_after": 1, "rows_before": 7, "rows_after": 4}
    retained = [pl.read_parquet(fp) for fp in files]
    assert [df.height for df in retained] == [2, 2]
    assert sorted(tuple(row) for df in retained for row in df["answers"].to_list()) == [
        (False, False),
        (False, False),
        (False, True),
        (False, True),
    ]
    assert all("prior_answers" not in df.columns for df in retained)


def test_generator_filter_regenerates_when_threshold_changes_or_pass_is_interrupted(
    tmp_path: Path, data_dir: Path, codes_yaml: Path
):
    out_dir = tmp_path / "grid"
    spec = tmp_path / "spec.yaml"
    code = yaml.safe_load(codes_yaml.read_text())[0]
    spec.write_text(yaml.safe_dump({"common": [entry(code, 200)]}))
    args = {"sequences_path": spec, "prediction_times_per_subject": 5}

    _run_seq(data_dir, out_dir, codes_yaml, **args)
    n_positive = sum(_labels(out_dir, shard)["answers"].list.last().sum() for shard in ("0", "1"))
    assert n_positive >= 2
    raw_columns = _labels(out_dir, "0").columns

    _run_seq(data_dir, out_dir, codes_yaml, **args, min_task_positives=n_positive)
    original = [_labels(out_dir, shard) for shard in ("0", "1")]
    assert all(df.height > 0 for df in original)
    # The key's defaulted optional columns are not written back, and eval_unique mirrors eval/.
    assert all(df.columns == raw_columns for df in original)
    contexts = ["subject_id", "prediction_time"]
    for shard, df in zip(("0", "1"), original, strict=True):
        assert _unique(out_dir, shard).equals(df.select(contexts).unique().sort(contexts))
    for shard in ("0", "1"):
        QuerySeqSchema.align(pq.read_table(out_dir / "eval" / SPLIT / f"{shard}.parquet"))
    marker = eval_seq._support_filter_marker(out_dir, SPLIT)
    assert json.loads(marker.read_text())["summary"]["rows_after"] == sum(df.height for df in original)

    inodes = [(out_dir / "eval" / SPLIT / f"{shard}.parquet").stat().st_ino for shard in ("0", "1")]
    _run_seq(data_dir, out_dir, codes_yaml, **args, min_task_positives=n_positive)
    assert inodes == [(out_dir / "eval" / SPLIT / f"{shard}.parquet").stat().st_ino for shard in ("0", "1")]

    # Raising the threshold removes all tasks.  Lowering it must reconstruct the raw labels,
    # rather than recounting the now-empty outputs.
    _run_seq(data_dir, out_dir, codes_yaml, **args, min_task_positives=n_positive + 1)
    assert all(_labels(out_dir, shard).is_empty() for shard in ("0", "1"))
    assert all(_unique(out_dir, shard).is_empty() for shard in ("0", "1"))
    _run_seq(data_dir, out_dir, codes_yaml, **args, min_task_positives=n_positive)
    assert all(_labels(out_dir, shard).equals(raw) for shard, raw in zip(("0", "1"), original, strict=True))

    # Simulate a crash after rewriting only one shard: its sidecar still matches, but the missing
    # completion marker forces *all* shards to be relabeled before global support is recounted.
    partial_fp = out_dir / "eval" / SPLIT / "0.parquet"
    pl.read_parquet(partial_fp).clear().write_parquet(partial_fp)
    marker.unlink()
    _run_seq(data_dir, out_dir, codes_yaml, **args, min_task_positives=n_positive)
    assert marker.exists()
    assert all(_labels(out_dir, shard).equals(raw) for shard, raw in zip(("0", "1"), original, strict=True))
