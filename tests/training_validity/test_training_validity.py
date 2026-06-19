"""Training-validity E2E test — Design 2: oracle markers + fire-time patterns.

See #123 for the design space.  This variant uses deterministic per-subject markers at day 0 that uniquely
identify (a) when the target event will fire, and (b) when the subject's data ends.  The intent is to give the
model a noise-free mapping from history to labels so the learning task is tractable in CI budget and the
unified AUROC criteria can be assessed cleanly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from meds import DatasetMetadataSchema, train_split, tuning_split
from meds_testing_helpers.dataset import MEDSDataset

from conftest import ENSURE_ENV_PLACEHOLDERS, run_and_check

if TYPE_CHECKING:
    from pathlib import Path


# ── Dataset-design constants ──────────────────────────────────────────────

_TARGET_CODE = "TARGET"
_NOISE_CODES = [f"NOISE_{i}" for i in range(5)]
_NOISE_RATE_PER_DAY = 0.5

# Fire-time markers.  Per-subject marker at day 0 determines when (if at all) the single
# TARGET event fires.  Event time = prediction_time + offset, picked so each marker maps
# uniquely onto which durations the event falls inside.  Durations chosen to straddle the
# offsets so label rates span the full [0, 1] range across durations.
_PREDICTION_TIME_DAYS = 10
_DURATIONS_DAYS = [1, 7, 30]
# offset, marker_name, fires_in_duration
# window_end(d=1)=11, d=7=17, d=30=40
_FIRE_PATTERNS: dict[str, float | None] = {
    "P_FIRE_D05": 0.5,  # window_end 11 > 10.5 AND > pred_time=10 → fires at d=1, 7, 30
    "P_FIRE_D5": 5.0,  # fires at d=7, 30
    "P_FIRE_D15": 15.0,  # fires at d=30
    "P_FIRE_D50": 50.0,  # never (outside all tested windows; also past END_D20)
    "P_NEVER": None,
}

# End-of-data markers → max_time.
_END_MARKERS: dict[str, int] = {
    "P_END_D20": 20,
    "P_END_D100": 100,
}

_N_TRAIN_SUBJECTS = 100
_N_TUNING_SUBJECTS = 50
_BASE_TIME = datetime(2020, 1, 1, tzinfo=UTC)
_DATASET_SEED = 1


def _synthesize_meds(out_dir: Path, seed: int = _DATASET_SEED) -> dict[str, pl.DataFrame]:
    """Write a synthetic MEDS dataset with oracle markers + the one TARGET event + noise fills."""
    rng = np.random.default_rng(seed)

    fire_names = list(_FIRE_PATTERNS.keys())
    end_names = list(_END_MARKERS.keys())

    events_per_split: dict[str, pl.DataFrame] = {}
    splits_rows: list[dict] = []

    for split_name, n_subjects, subj_offset in [
        (train_split, _N_TRAIN_SUBJECTS, 1000),
        (tuning_split, _N_TUNING_SUBJECTS, 2000),
    ]:
        rows = []
        for subj_id in range(subj_offset, subj_offset + n_subjects):
            splits_rows.append({"subject_id": subj_id, "split": split_name})

            fire_marker = fire_names[int(rng.integers(0, len(fire_names)))]
            end_marker = end_names[int(rng.integers(0, len(end_names)))]
            fire_offset = _FIRE_PATTERNS[fire_marker]
            max_time_day = _END_MARKERS[end_marker]

            # Markers at day 0.
            for marker in (fire_marker, end_marker):
                rows.append(
                    {
                        "subject_id": subj_id,
                        "time": _BASE_TIME,
                        "code": marker,
                        "numeric_value": None,
                    }
                )

            # Single TARGET event, if the subject is a firing type AND the event falls within
            # the subject's observation window.
            if fire_offset is not None:
                event_day = _PREDICTION_TIME_DAYS + fire_offset
                if event_day < max_time_day:
                    rows.append(
                        {
                            "subject_id": subj_id,
                            "time": _BASE_TIME + timedelta(days=event_day),
                            "code": _TARGET_CODE,
                            "numeric_value": None,
                        }
                    )

            # Noise events — Poisson per noise code from day 0.01 (after the markers) to
            # max_time_day.  Makes the sequence non-trivial so the model has to attend to the
            # markers rather than treating every history as identical.
            for code in _NOISE_CODES:
                n_events = int(rng.poisson(_NOISE_RATE_PER_DAY * max_time_day))
                times_days = np.sort(rng.uniform(0.01, float(max_time_day), size=n_events))
                for t_days in times_days:
                    rows.append(
                        {
                            "subject_id": subj_id,
                            "time": _BASE_TIME + timedelta(days=float(t_days)),
                            "code": code,
                            "numeric_value": None,
                        }
                    )

        events_per_split[split_name] = pl.DataFrame(
            rows,
            schema={
                "subject_id": pl.Int64,
                "time": pl.Datetime("us", "UTC"),
                "code": pl.Utf8,
                "numeric_value": pl.Float64,
            },
        ).sort(["subject_id", "time", "code"])  # `code` breaks ties between the two day-0 markers

    subject_splits = pl.DataFrame(splits_rows, schema={"subject_id": pl.Int64, "split": pl.Utf8})
    all_codes = [_TARGET_CODE, *fire_names, *end_names, *_NOISE_CODES]
    code_metadata = pl.DataFrame({"code": all_codes}, schema={"code": pl.Utf8})

    MEDSDataset(
        data_shards={f"{split}/0": df for split, df in events_per_split.items()},
        dataset_metadata=DatasetMetadataSchema(dataset_name="d2_synthetic", dataset_version="0.1"),
        code_metadata=code_metadata,
        subject_splits=subject_splits,
    ).write(out_dir)

    return events_per_split


def _compute_labels(events: pl.DataFrame, subject_ids: list[int]) -> pl.DataFrame:
    """Per-(subject, duration) labels via ``evaluate_index_df`` semantics — one code (TARGET)."""
    window_start = _BASE_TIME + timedelta(days=_PREDICTION_TIME_DAYS)
    max_time_per_subject = dict(
        events.group_by("subject_id").agg(pl.col("time").max().alias("max_time")).iter_rows()
    )

    rows = []
    for subj in subject_ids:
        max_time = max_time_per_subject[subj]
        subj_events = events.filter(pl.col("subject_id") == subj)
        for duration_days in _DURATIONS_DAYS:
            window_end = _BASE_TIME + timedelta(days=float(_PREDICTION_TIME_DAYS + duration_days))
            censored = window_end > max_time
            event_fires = not subj_events.filter(
                (pl.col("code") == _TARGET_CODE)
                & (pl.col("time") > window_start)
                & (pl.col("time") <= window_end)
            ).is_empty()
            # Collapsed nullable boolean_value per TaskQuerySchema:
            #   null  → censored
            #   True  → event occurred in window
            #   False → no event, not censored
            boolean_value = None if censored else event_fires
            rows.append(
                {
                    "subject_id": subj,
                    "prediction_time": window_start,
                    "boolean_value": boolean_value,
                    "query": _TARGET_CODE,
                    "duration_days": duration_days,
                }
            )

    return pl.DataFrame(
        rows,
        schema={
            "subject_id": pl.Int64,
            "prediction_time": pl.Datetime("us", "UTC"),
            "boolean_value": pl.Boolean,
            "query": pl.Utf8,
            "duration_days": pl.Int64,
        },
    ).with_columns(pl.col("prediction_time").dt.replace_time_zone(None))


# `oracle_dataset` is defined in `tests/training_validity/conftest.py` (session-scoped)
# so the README doctest and this test module share one synthesis run + byte-identical
# ``events`` / ``labels`` DataFrames.  It's auto-discovered here via pytest's normal
# conftest lookup — no import needed.


@pytest.fixture(scope="module")
def oracle_preprocessed(oracle_dataset: dict, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("d2_preprocessed")
    intermediate = root / "intermediate"
    final = root / "final"
    run_and_check(
        [
            "EQ_process_data",
            f"input_dir={oracle_dataset['meds_dir']!s}",
            f"intermediate_dir={intermediate!s}",
            f"output_dir={final!s}",
            "do_demo=True",
        ],
        timeout=300.0,
    )
    return final


@pytest.fixture(scope="module")
def oracle_trained_model_dir(
    oracle_preprocessed: Path,
    oracle_dataset: dict,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    output_dir = tmp_path_factory.mktemp("d2_train_out")
    run_and_check(
        [
            "EQ_train",
            "--config-name=_demo_train",
            f"output_dir={output_dir!s}",
            f"datamodule.config.tensorized_cohort_dir={oracle_preprocessed!s}",
            f"datamodule.config.task_labels_dir={oracle_dataset['task_labels_dir']!s}",
            "datamodule.batch_size=16",
            "datamodule.config.max_seq_len=128",
            "lightning_module.model.config_overrides.hidden_size=128",
            "lightning_module.model.config_overrides.num_hidden_layers=4",
            "lightning_module.model.config_overrides.num_attention_heads=4",
            "lightning_module.model.config_overrides.intermediate_size=256",
            "lightning_module.model.config_overrides.max_position_embeddings=512",
            # 2000 steps is enough once the weight init is reproducible — the #124 fix moved
            # `seed_everything` above `instantiate(cfg.lightning_module)`, so weight init is
            # now platform-independent for a given `cfg.seed`.  Before that, an unlucky 3.12
            # init had the censor head stuck at AUROC 0.765 / flat duration means at 2000
            # steps; that can't happen anymore.
            "trainer.max_steps=2000",
            "trainer.max_epochs=10000",
            "trainer.limit_val_batches=2",
            "trainer.val_check_interval=1000",
            "lightning_module.optimizer.lr=1e-3",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=1800.0,
    )
    return output_dir


# Unified passing criteria (applied across every design in #123).
_CENSOR_AUROC_THRESHOLD = 0.9
_PER_CELL_OCCURS_AUROC_THRESHOLD = 0.8


@pytest.mark.slow
def test_trained_model_learns_occurs_and_censor(
    oracle_preprocessed: Path,
    oracle_dataset: dict,
    oracle_trained_model_dir: Path,
    tmp_path: Path,
) -> None:
    """Unified criteria: censor AUROC >= 0.9, per-(code, duration) occurs AUROC >= 0.8 on every
    non-degenerate cell, and per-code mean predictions strictly increase with duration.

    Collects all per-cell failures before asserting, so the error message shows the full cell
    matrix rather than short-circuiting on the first miss.

    Marked ``slow`` and skipped by default (see ``pyproject.toml`` ``addopts``).  Opt in with
    ``pytest -m slow`` (run only this) or ``pytest -m 'slow or not slow'`` (run everything).
    CI's `tests.yaml` uses the latter.

    Both stages — inference (``EQ_predict``) and metrics (``EQ_evaluate``) — run
    as subprocesses against actual CLI outputs, so the unified criteria land on
    what a caller would get in production, not an in-process recomputation.
    Predictions go against the tuning split (that's where this test's synthetic
    validation data lives).
    """
    # ── Stage 1: EQ_predict → PredictionSchema parquet ──────────────────────
    predictions_parquet = tmp_path / "predictions.parquet"
    tuning_tasks_dir = oracle_dataset["task_labels_dir"] / tuning_split
    run_and_check(
        [
            "EQ_predict",
            f"model_run_dir={oracle_trained_model_dir!s}",
            f"tasks_dir={tuning_tasks_dir!s}",
            f"output_parquet={predictions_parquet!s}",
            f"split={tuning_split}",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=600.0,
    )

    # ── Stage 2: EQ_evaluate → metrics parquet ──────────────────────────────
    metrics_parquet = tmp_path / "metrics.parquet"
    run_and_check(
        [
            "EQ_evaluate",
            f"predictions_parquet={predictions_parquet!s}",
            f"metrics_parquet={metrics_parquet!s}",
        ],
        env=ENSURE_ENV_PLACEHOLDERS,
        timeout=60.0,
    )
    metrics = pl.read_parquet(metrics_parquet).sort(["query", "duration_days"])
    # Full diagnostic dump regardless of pass/fail.
    print("\n[Design 2] per-(query, duration_days) metrics (tuning):")
    for row in metrics.iter_rows(named=True):
        print(
            f"  {row['query']:8s} d={int(row['duration_days']):3d}  "
            f"n={row['n_rows']:3d} labeled={row['n_occurs_labeled']:3d} "
            f"pos={row['n_positive']:3d}  "
            f"occurs_auroc={row['occurs_auroc']}  censor_auroc={row['censor_auroc']}"
        )

    # ── Stage 3: assertions on the metrics parquet + monotonicity on preds ──
    failures: list[str] = []

    # 1. Per-cell `occurs_auroc` on non-censored rows must meet threshold.  Every
    #    (TARGET, duration) cell is expected to have both classes after filtering
    #    out censored subjects (see the windowing table in README.md) — a null
    #    occurs_auroc from `evaluate.evaluate` means the cell went single-class,
    #    which is a synthesis/labeling regression.
    for row in metrics.filter(pl.col("query") == _TARGET_CODE).iter_rows(named=True):
        d = int(row["duration_days"])
        if row["occurs_auroc"] is None:
            failures.append(
                f"occurs_auroc is null at (TARGET, d={d}), n_labeled={row['n_occurs_labeled']}, "
                f"n_positive={row['n_positive']}.  Expected both occurs classes from "
                f"firing+non-firing markers (see README windowing table)."
            )
            continue
        if row["occurs_auroc"] < _PER_CELL_OCCURS_AUROC_THRESHOLD:
            failures.append(
                f"per-cell occurs_auroc (TARGET, d={d}) is {row['occurs_auroc']:.3f} < "
                f"{_PER_CELL_OCCURS_AUROC_THRESHOLD}"
            )

    # 2. Censor head — only d=30 has mixed censor labels in Design 2 (P_END_D20's
    #    max_time=20 < window_end=40; P_END_D100's doesn't).  d=1 and d=7 are
    #    single-class-censor by construction, so `evaluate.evaluate` reports null
    #    censor_auroc there.  Assert the threshold on the one defined cell.
    censor_cells = metrics.filter(pl.col("censor_auroc").is_not_null())
    if censor_cells.is_empty():
        failures.append(
            "censor_auroc is null for every (query, duration_days) cell — expected "
            "at least d=30 to mix P_END_D20 (censored) and P_END_D100 (not censored)."
        )
    else:
        for row in censor_cells.iter_rows(named=True):
            if row["censor_auroc"] < _CENSOR_AUROC_THRESHOLD:
                failures.append(
                    f"censor_auroc (query={row['query']}, d={int(row['duration_days'])}) is "
                    f"{row['censor_auroc']:.3f} < {_CENSOR_AUROC_THRESHOLD}"
                )

    # 3. Duration monotonicity — the occurs head is loss-masked on censored samples
    #    (model.py: ``occurs_loss(..., mask=~batch.censor)``), so its predictions on
    #    those rows are unconstrained and would add noise to the mean.  Not a per-cell
    #    metric — lives on the predictions parquet rather than the metrics parquet.
    predictions = pl.read_parquet(predictions_parquet)
    non_censored = predictions.filter(pl.col("boolean_value").is_not_null())
    means = [
        float(non_censored.filter(pl.col("duration_days") == float(d))["occurs_prob"].mean())
        for d in _DURATIONS_DAYS
    ]
    mean_map = dict(zip(_DURATIONS_DAYS, [round(m, 3) for m in means], strict=False))
    print(f"\n  duration-mean occurs_prob (TARGET, non-censored): {mean_map}")
    for d1, d2, m1, m2 in zip(_DURATIONS_DAYS[:-1], _DURATIONS_DAYS[1:], means[:-1], means[1:], strict=False):
        if m2 <= m1:
            failures.append(
                f"duration monotonicity violated: mean pred at d={d2} ({m2:.3f}) not > d={d1} ({m1:.3f})"
            )

    if failures:
        raise AssertionError("\n".join(failures))
