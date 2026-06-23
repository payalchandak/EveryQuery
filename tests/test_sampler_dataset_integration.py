"""Downstream integration: sampler output is a drop-in for ``EveryQueryPytorchDataset``.

This lives at the top level (not under ``tests/sampler/``) on purpose: it depends on ``demo_model``
and ``tensorized_cohort_dir`` from the repo-root ``conftest.py``, so it needs the root autouse
fixtures (and HuggingFace/network access to build the ModernBERT).  The pure, offline sampler-stage
unit tests live in ``tests/sampler/``.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import torch
from meds import train_split
from meds_torchdata.config import MEDSTorchDataConfig

from every_query.data.dataset import EveryQueryPytorchDataset
from every_query.data.schema import TaskQuerySchema
from every_query.generate_tasks.sample_tasks import evaluate_index_df

# Subject IDs and prediction times from conftest / simple_static_MEDS.  Duplicated here rather
# than imported to keep the test module self-contained.
_E2E_TRAIN_SUBJECTS = [239684, 1195293, 68729, 814703]
_E2E_PRED_TIMES: dict[int, datetime] = {
    239684: datetime(2010, 5, 11, 18, 0, tzinfo=UTC),
    1195293: datetime(2010, 6, 20, 20, 30, tzinfo=UTC),
    68729: datetime(2010, 5, 26, 3, 0, tzinfo=UTC),
    814703: datetime(2010, 2, 5, 6, 0, tzinfo=UTC),
}
_E2E_QUERIES = ["HR", "TEMP"]


class TestEndToEndWithDataset:
    """Sanity-check that sampler output is a drop-in replacement for the dataset's task-labels contract.

    We don't run the full ``run()`` pipeline here because its shard-reading path expects string-coded
    events from an early preprocessing stage that this fixture does not expose (the tensorized cohort
    only materializes ``normalization/*.parquet`` with *integer-coded* events downstream of
    tokenization).  ``tests/sampler/test_orchestration.py`` already exercises the full pipeline on a
    synthetic cohort.

    Instead, we test the downstream half of the pipeline: hand-build an ``index_df`` referencing
    the real subject IDs in ``tensorized_cohort_dir``, call ``evaluate_index_df`` on it against a
    synthetic events table, write the result as sampler-shaped parquet, then instantiate
    ``EveryQueryPytorchDataset`` against the real tokenized shards + the sampler-shaped labels and
    push a collated batch through ``demo_model``.  This is the minimum sufficient test to catch
    schema drift between the sampler's output and the downstream dataset.
    """

    def test_sampler_output_is_drop_in_for_dataset(self, tensorized_cohort_dir, tmp_path_factory, demo_model):
        labels_dir = tmp_path_factory.mktemp("sampler_labels")

        # Build a tiny events df covering the real train subjects.  Each subject gets an "HR"
        # event at its prediction_time plus a "TEMP" event ~100 days later; the later event
        # pushes max_time comfortably past prediction_time + 30d so the resulting rows are
        # *uncensored* (otherwise the whole batch would be censored and the occurs-loss BCE on
        # an empty mask would NaN out — see issue #30).
        rows = []
        for subj in _E2E_TRAIN_SUBJECTS:
            pt = _E2E_PRED_TIMES[subj]
            rows.append({"subject_id": subj, "time": pt, "code": "HR"})
            rows.append({"subject_id": subj, "time": pt + timedelta(days=5), "code": "HR"})
            rows.append({"subject_id": subj, "time": pt + timedelta(days=100), "code": "TEMP"})
        events_df = pl.DataFrame(rows).cast({"time": pl.Datetime("us")})

        # Index_df: one row per (subject x query), fixed duration.  This is the shape sampler.py
        # would produce for n_tasks=len(_E2E_QUERIES), contexts_per_task=len(_E2E_TRAIN_SUBJECTS)
        # with specific draws; constructing it directly bypasses the sampler randomness.
        index_df = pl.DataFrame(
            [
                {
                    "subject_id": subj,
                    "prediction_time": _E2E_PRED_TIMES[subj],
                    "query": q,
                    "duration_days": 30,
                }
                for subj in _E2E_TRAIN_SUBJECTS
                for q in _E2E_QUERIES
            ]
        ).cast(
            {
                "subject_id": pl.Int64,
                "prediction_time": pl.Datetime("us"),
                "query": pl.Utf8,
                # Float32 to match TaskQuerySchema.duration_days.
                "duration_days": pl.Float32,
            }
        )

        labeled = evaluate_index_df(index_df, events_df)

        # Sanity: labeled output conforms to TaskQuerySchema — same check the Stage 4 write path
        # (``label_one_shard``) does, exercised here at the per-function boundary.
        TaskQuerySchema.validate(labeled.to_arrow())

        labels_fp = Path(labels_dir) / "0.parquet"
        labeled.write_parquet(labels_fp)

        cfg = MEDSTorchDataConfig(
            tensorized_cohort_dir=str(tensorized_cohort_dir),
            task_labels_dir=str(labels_dir),
            max_seq_len=64,
            seq_sampling_strategy="to_end",
            static_inclusion_mode="omit",
            batch_mode="SM",
        )
        dataset = EveryQueryPytorchDataset(cfg, split=train_split)
        assert len(dataset) >= 2

        items = [dataset[0], dataset[1]]
        batch = dataset.collate(items)

        with torch.no_grad():
            loss, out = demo_model._forward(batch)

        assert torch.isfinite(loss).item(), f"Non-finite loss on sampler-produced batch: {loss}"
        assert out.query_embed.shape[0] == batch.batch_size
