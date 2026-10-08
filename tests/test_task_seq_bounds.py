"""Unit tests for the ``get_task_seq_bounds_and_labels`` overrides that hstack extra label columns.

Guards against silent misalignment between the upstream-computed end indices and the
EveryQuery-specific annotation columns that each subclass hstacks onto the result:

* ``EveryQueryPytorchDataset`` — the single-query columns (``occurs``, ``query``,
  ``duration_days``);
* ``QuerySeqPytorchDataset`` — the query-sequence list columns (``queries``, ``durations``,
  ``answers``, and the optional ``bound_events`` / ``start_*`` columns).

Both overrides rebuild the extras from a *semi-filtered copy of* ``label_df`` rather than from the
join result, which is only sound while upstream preserves ``label_df`` input order for surviving
rows.  The ``QuerySeqPytorchDataset`` half below is the sibling of the pair covering
``MultitaskBoundaryPytorchDataset`` in ``tests/test_multitask_dataset_integration.py``.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from meds import DataSchema, LabelSchema

from every_query.data.dataset import EveryQueryPytorchDataset
from every_query.data.query_seq_dataset import QuerySeqPytorchDataset


def _schema_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            DataSchema.subject_id_name: [1, 2, 3],
            DataSchema.time_name: [
                [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
                [datetime(2020, 1, 1), datetime(2020, 1, 5)],
                [datetime(2020, 1, 10)],
            ],
        }
    )


class TestExtraColumnAlignment:
    def test_extras_align_to_original_label_rows(self):
        label_df = pl.DataFrame(
            {
                DataSchema.subject_id_name: [1, 1, 2, 3, 99],
                LabelSchema.prediction_time_name: [
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 5),
                    datetime(2020, 1, 10),
                    datetime(2020, 1, 1),
                ],
                "boolean_value": [True, False, True, False, True],
                "occurs": [10, 20, 30, 40, 50],
                "query": [101, 102, 103, 104, 105],
                "duration_days": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = EveryQueryPytorchDataset.get_task_seq_bounds_and_labels(label_df, _schema_df())

        # Subject 99 is absent from schema_df → inner-join semantics drop it.
        assert result.height == 4
        assert set(result.columns) >= {
            DataSchema.subject_id_name,
            LabelSchema.prediction_time_name,
            EveryQueryPytorchDataset.END_IDX,
            "boolean_value",
            "occurs",
            "query",
            "duration_days",
        }

        # For each surviving row, the extras must match the row in label_df with the
        # same (subject_id, prediction_time).
        label_lookup = {
            (row[DataSchema.subject_id_name], row[LabelSchema.prediction_time_name]): row
            for row in label_df.iter_rows(named=True)
        }
        for out_row in result.iter_rows(named=True):
            key = (out_row[DataSchema.subject_id_name], out_row[LabelSchema.prediction_time_name])
            expected = label_lookup[key]
            assert out_row["occurs"] == expected["occurs"]
            assert out_row["query"] == expected["query"]
            assert out_row["duration_days"] == expected["duration_days"]
            assert out_row["boolean_value"] == expected["boolean_value"]

    def test_no_extras_passes_through(self):
        label_df = pl.DataFrame(
            {
                DataSchema.subject_id_name: [1, 2],
                LabelSchema.prediction_time_name: [datetime(2020, 1, 2), datetime(2020, 1, 5)],
                "boolean_value": [True, False],
            }
        )

        result = EveryQueryPytorchDataset.get_task_seq_bounds_and_labels(label_df, _schema_df())

        assert result.height == 2
        assert "occurs" not in result.columns
        assert "query" not in result.columns
        assert "duration_days" not in result.columns

    def test_partial_extras(self):
        label_df = pl.DataFrame(
            {
                DataSchema.subject_id_name: [1, 2],
                LabelSchema.prediction_time_name: [datetime(2020, 1, 2), datetime(2020, 1, 5)],
                "boolean_value": [True, False],
                "query": [7, 8],
            }
        )

        result = EveryQueryPytorchDataset.get_task_seq_bounds_and_labels(label_df, _schema_df())

        assert "query" in result.columns
        assert "occurs" not in result.columns
        assert "duration_days" not in result.columns
        by_sid = {r[DataSchema.subject_id_name]: r["query"] for r in result.iter_rows(named=True)}
        assert by_sid == {1: 7, 2: 8}


# --- QuerySeqPytorchDataset: the same override over query-sequence list columns ------------------


def _adversarial_seq_label_frames() -> tuple[pl.DataFrame, pl.DataFrame, list[int]]:
    """A shuffled query-sequence ``label_df`` with foreign subjects, plus the ``schema_df`` it joins.

    The adversarial construction is the whole point: the subject order is scrambled *and*
    subjects 97/98/99 (absent from ``schema_df``) are interleaved between the surviving rows, so a
    guard that has stopped working shows up as a shift.  A sorted or single-subject frame would
    pass whether or not the alignment check does anything.  ``queries`` carries the input row
    index (``Q{i}``), so the assertions can name the exact row that moved.

    Returns the frames plus the label rows (by input position) that survive the join.
    """
    known = [3, 1, 2]
    base = datetime(2020, 1, 1)  # naive timestamps are fine for synthetic frames
    rows = []
    for i, sid in enumerate([2, 99, 3, 1, 3, 98, 2, 1, 97]):
        rows.append(
            {
                DataSchema.subject_id_name: sid,
                LabelSchema.prediction_time_name: base + timedelta(days=(7 * i) % 30),
                "queries": [f"Q{i}"],
                "durations": [float(i)],
                "answers": [i % 2 == 0],
            }
        )
    label_df = pl.DataFrame(rows).cast(
        {
            DataSchema.subject_id_name: pl.Int64,
            LabelSchema.prediction_time_name: pl.Datetime("us"),
            "queries": pl.List(pl.Utf8),
            "durations": pl.List(pl.Float32),
            "answers": pl.List(pl.Boolean),
        }
    )
    schema_df = pl.DataFrame(
        {
            DataSchema.subject_id_name: known,
            DataSchema.time_name: [[base + timedelta(days=d) for d in range(0, 40, 5)]] * len(known),
        }
    ).cast({DataSchema.subject_id_name: pl.Int64})
    surviving = [i for i, r in enumerate(rows) if r[DataSchema.subject_id_name] in known]
    return label_df, schema_df, surviving


def test_seq_label_extras_stay_aligned_with_the_upstream_rows() -> None:
    """Foreign subjects and a shuffled input order must not shift the hstacked query columns.

    The list columns are hstacked positionally onto the upstream bounds frame, so any shift pairs one
    context's queries with another context's patient window — wrong labels, no error.
    """
    label_df, schema_df, surviving = _adversarial_seq_label_frames()
    out = QuerySeqPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)
    assert out["queries"].to_list() == label_df["queries"][surviving].to_list()
    assert (
        out[DataSchema.subject_id_name].to_list() == label_df[DataSchema.subject_id_name][surviving].to_list()
    )
    assert (
        out[LabelSchema.prediction_time_name].to_list()
        == label_df[LabelSchema.prediction_time_name][surviving].to_list()
    )
    assert out["durations"].to_list() == label_df["durations"][surviving].to_list()
    assert out["answers"].to_list() == label_df["answers"][surviving].to_list()


def test_seq_upstream_reordering_is_an_error_not_a_silent_misalignment(monkeypatch) -> None:
    """If a polars/MTD bump ever stops preserving label order, the hstack must fail loudly.

    ``maintain_order="left"`` pins *our* side of the semi join; nothing pins upstream's. The
    ``_check_rows_aligned`` call is what turns that drift into a RuntimeError rather than silently
    mislabelled training data.
    """
    from meds_torchdata import MEDSPytorchDataset

    label_df, schema_df, _ = _adversarial_seq_label_frames()
    real = MEDSPytorchDataset.get_task_seq_bounds_and_labels.__func__

    def reversed_upstream(cls, l_df, s_df):
        return real(cls, l_df, s_df).reverse()

    monkeypatch.setattr(MEDSPytorchDataset, "get_task_seq_bounds_and_labels", classmethod(reversed_upstream))
    with pytest.raises(RuntimeError, match=r"misaligned .* row\(s\) differ"):
        QuerySeqPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)
