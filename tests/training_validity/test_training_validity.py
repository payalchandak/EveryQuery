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
import torch
from meds import DatasetMetadataSchema, train_split, tuning_split
from meds_testing_helpers.dataset import MEDSDataset
from meds_torchdata.config import MEDSTorchDataConfig
from sklearn.metrics import roc_auc_score

from conftest import ENSURE_ENV_PLACEHOLDERS, run_and_check
from every_query.data.dataset import EveryQueryPytorchDataset
from every_query.model.lightning_module import EveryQueryLightningModule

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
                & (pl.col("time") < window_end)
            ).is_empty()
            occurs = (not censored) and event_fires
            rows.append(
                {
                    "subject_id": subj,
                    "prediction_time": window_start,
                    "boolean_value": censored,
                    "occurs": occurs,
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
            "occurs": pl.Boolean,
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
            f"query.codes=[{_TARGET_CODE}]",
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


def _load_lightning_module(trained_model_dir: Path) -> EveryQueryLightningModule:
    """Load the monitor-best checkpoint that ``train.py`` copies to ``best_model.ckpt``.

    ``train.py`` selects the best checkpoint by ``tuning/loss`` (``ModelCheckpoint(monitor=...,
    save_top_k=1)``) and copies it to ``output_dir/best_model.ckpt``; loading that file
    directly matches the training contract and avoids the ``last.ckpt`` vs.
    ``epoch=N-step=M.ckpt`` ambiguity.
    """
    best_path = trained_model_dir / "best_model.ckpt"
    assert best_path.exists(), (
        f"best_model.ckpt missing under {trained_model_dir} — EQ_train's post-training copy "
        f"(train.py ~line 272) didn't happen; training likely crashed before the best-ckpt "
        f"save.  Check the subprocess stderr in the run_and_check log."
    )
    module = EveryQueryLightningModule.load_from_checkpoint(str(best_path))
    module.eval()
    return module


# Unified passing criteria (applied across every design in #123).
_CENSOR_AUROC_THRESHOLD = 0.9
_PER_CELL_OCCURS_AUROC_THRESHOLD = 0.8


@pytest.mark.slow
def test_trained_model_learns_occurs_and_censor(
    oracle_preprocessed: Path,
    oracle_dataset: dict,
    oracle_trained_model_dir: Path,
) -> None:
    """Unified criteria: censor AUROC >= 0.9, per-(code, duration) occurs AUROC >= 0.8 on every
    non-degenerate cell, and per-code mean predictions strictly increase with duration.

    Collects all per-cell failures before asserting, so the error message shows the full cell
    matrix rather than short-circuiting on the first miss.

    Marked ``slow`` and skipped by default (see ``pyproject.toml`` ``addopts``).  Opt in with
    ``pytest -m slow`` (run only this) or ``pytest -m 'slow or not slow'`` (run everything).
    CI's `tests.yaml` uses the latter.

    NOTE: inference runs in-process rather than via a ``EQ_predict`` CLI subprocess because the
    inference CLI (``feat/eq-predict``, PR #99) is still a draft pending design-doc sign-off
    on #81.  Once #99 merges, migrate this to a subprocess ``EQ_predict`` call + a
    ``PredictionSchema`` parquet read, matching the pattern of the other E2E tests.
    """
    module = _load_lightning_module(oracle_trained_model_dir)

    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=str(oracle_preprocessed),
        task_labels_dir=str(oracle_dataset["task_labels_dir"]),
        max_seq_len=128,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )
    dataset = EveryQueryPytorchDataset(cfg, split=tuning_split)

    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        batch = dataset.collate([item])
        with torch.no_grad():
            _, outputs = module.model(batch)
        rows.append(
            {
                "query": dataset.query[i],
                "duration_days": int(dataset.duration_days[i]),
                "occurs_true": int(item["occurs"]),
                "censor_true": int(item["boolean_value"]),
                "occurs_pred": float(outputs.occurs_probs.squeeze().item()),
                "censor_pred": float(outputs.censor_probs.squeeze().item()),
            }
        )
    df = pl.DataFrame(rows)

    # Diagnostics — always print, regardless of pass/fail.
    print("\n[Design 2] per-cell label + prediction distribution (tuning):")
    per_cell = (
        df.group_by(["query", "duration_days"])
        .agg(
            pl.len().alias("n"),
            pl.col("occurs_true").cast(pl.Float64).mean().alias("p_occurs_true"),
            pl.col("occurs_pred").mean().alias("p_occurs_pred"),
            pl.col("censor_true").cast(pl.Float64).mean().alias("p_censor_true"),
            pl.col("censor_pred").mean().alias("p_censor_pred"),
        )
        .sort(["query", "duration_days"])
    )
    for r in per_cell.iter_rows(named=True):
        print(
            f"  {r['query']:8s} d={r['duration_days']:3d}  n={r['n']:3d}  "
            f"p(occurs)={r['p_occurs_true']:.3f} (pred {r['p_occurs_pred']:.3f})  "
            f"p(censor)={r['p_censor_true']:.3f} (pred {r['p_censor_pred']:.3f})"
        )

    failures: list[str] = []

    # ── 1. Censor-head AUROC ─────────────────────────────────────────────────
    # Censoring is part of the required signal in Design 2 (P_END_D20 guarantees censoring at
    # d=30), so a single-class censor distribution is itself a regression — fail explicitly
    # rather than silently skipping the AUROC check.
    censor_true = df["censor_true"].to_list()
    censor_pred = df["censor_pred"].to_list()
    censor_classes = set(censor_true)
    if len(censor_classes) < 2:
        n_pos = sum(censor_true)
        failures.append(
            f"censor label distribution is degenerate — {len(censor_true)} rows, "
            f"{n_pos} censored, classes={sorted(censor_classes)}.  Expected both classes "
            f"(P_END_D20 subjects should be censored at d=30, P_END_D100 should not)."
        )
    else:
        censor_auroc = float(roc_auc_score(censor_true, censor_pred))
        print(f"\ncensor-head AUROC: {censor_auroc:.3f}")
        if censor_auroc < _CENSOR_AUROC_THRESHOLD:
            failures.append(f"censor AUROC {censor_auroc:.3f} < {_CENSOR_AUROC_THRESHOLD}")

    # ── 2. Per-(code, duration) occurs AUROC on non-censored rows ────────────
    # Each (TARGET, duration) cell is expected to have both occurs=True and occurs=False
    # after filtering to non-censored subjects (see the windowing table in README.md).  A
    # degenerate cell means synthesis/labeling regressed, so fail explicitly.
    for duration in _DURATIONS_DAYS:
        cell = df.filter(
            (pl.col("query") == _TARGET_CODE)
            & (pl.col("duration_days") == duration)
            & (pl.col("censor_true") == 0)
        )
        y_true = cell["occurs_true"].to_list()
        y_pred = cell["occurs_pred"].to_list()
        cell_classes = set(y_true)
        if len(cell_classes) < 2:
            failures.append(
                f"occurs label distribution is degenerate at (TARGET, d={duration}): "
                f"n={len(y_true)}, classes={sorted(cell_classes)}.  Expected both classes "
                f"from firing+non-firing markers (see README windowing table)."
            )
            continue
        auroc = float(roc_auc_score(y_true, y_pred))
        print(f"  occurs AUROC (TARGET, d={duration}, n={len(y_true)}): {auroc:.3f}")
        if auroc < _PER_CELL_OCCURS_AUROC_THRESHOLD:
            failures.append(
                f"per-cell AUROC (TARGET, d={duration}) is {auroc:.3f} < {_PER_CELL_OCCURS_AUROC_THRESHOLD}"
            )

    # ── 3. Duration monotonicity ─────────────────────────────────────────────
    # Averages over non-censored rows only — the occurs head is loss-masked on censored
    # samples (model.py: `occurs_loss(..., mask=~batch.censor)`), so its predictions on
    # those rows are unconstrained and would add noise to the mean.
    non_censored = df.filter(pl.col("censor_true") == 0)
    means = [
        float(non_censored.filter(pl.col("duration_days") == d)["occurs_pred"].mean())
        for d in _DURATIONS_DAYS
    ]
    mean_map = dict(zip(_DURATIONS_DAYS, [round(m, 3) for m in means], strict=False))
    print(f"\n  duration-mean occurs_pred (TARGET, non-censored): {mean_map}")
    for d1, d2, m1, m2 in zip(_DURATIONS_DAYS[:-1], _DURATIONS_DAYS[1:], means[:-1], means[1:], strict=False):
        if m2 <= m1:
            failures.append(
                f"duration monotonicity violated: mean pred at d={d2} ({m2:.3f}) not > d={d1} ({m1:.3f})"
            )

    if failures:
        raise AssertionError("\n".join(failures))
