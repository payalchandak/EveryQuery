"""Unit tests for ``EveryQueryPytorchDataset.get_task_seq_bounds_and_labels``.

Guards against silent misalignment between the upstream-computed end indices
and the EveryQuery-specific annotation columns (``occurs``, ``query``,
``duration_days``) that the subclass hstacks onto the result.
"""

from datetime import datetime

import polars as pl
from meds import DataSchema, LabelSchema

from every_query.data.dataset import EveryQueryPytorchDataset


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

    def test_extras_align_when_subject_ids_are_unsorted(self):
        """Subject ids deliberately out of sorted order (#299).

        The upstream implementation sorts by ``(subject_id, prediction_time)`` internally
        before restoring input order, and the subclass's own semi-join carries no row-order
        guarantee from polars. Both effects are invisible when the input happens to be sorted,
        so this input is shuffled and every row's annotations are distinct.
        """
        label_df = pl.DataFrame(
            {
                DataSchema.subject_id_name: [3, 1, 2, 1, 99, 2],
                LabelSchema.prediction_time_name: [
                    datetime(2020, 1, 10),
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 5),
                ],
                "boolean_value": [True, False, True, True, False, False],
                "occurs": [30, 12, 21, 11, 99, 22],
                "query": [300, 120, 210, 110, 990, 220],
                "duration_days": [3.0, 1.2, 2.1, 1.1, 9.9, 2.2],
            }
        )

        result = EveryQueryPytorchDataset.get_task_seq_bounds_and_labels(label_df, _schema_df())

        # Subject 99 is absent from schema_df → dropped.
        assert result.height == 5

        label_lookup = {
            (row[DataSchema.subject_id_name], row[LabelSchema.prediction_time_name]): row
            for row in label_df.iter_rows(named=True)
        }
        for out_row in result.iter_rows(named=True):
            key = (out_row[DataSchema.subject_id_name], out_row[LabelSchema.prediction_time_name])
            expected = label_lookup[key]
            assert out_row["occurs"] == expected["occurs"], f"occurs misaligned for {key}"
            assert out_row["query"] == expected["query"], f"query misaligned for {key}"
            assert out_row["duration_days"] == expected["duration_days"], f"duration misaligned for {key}"
            assert out_row["boolean_value"] == expected["boolean_value"], f"label misaligned for {key}"

        # End indices are still the ones the (subject_id, prediction_time) pair implies.
        end_idx = {
            (r[DataSchema.subject_id_name], r[LabelSchema.prediction_time_name]): r[
                EveryQueryPytorchDataset.END_IDX
            ]
            for r in result.iter_rows(named=True)
        }
        assert end_idx == {
            (3, datetime(2020, 1, 10)): 1,
            (1, datetime(2020, 1, 3)): 3,
            (2, datetime(2020, 1, 1)): 1,
            (1, datetime(2020, 1, 2)): 2,
            (2, datetime(2020, 1, 5)): 2,
        }

    def test_extras_align_across_repeated_subject_prediction_time(self):
        """Several query rows share one ``(subject_id, prediction_time)`` pair.

        That pair is therefore not a unique key, so alignment cannot be re-derived by a key
        join — it has to be positionally correct. Within each repeated group the rows are
        distinguishable only by ``boolean_value``, so the annotations are built to encode it
        (``occurs`` is even iff ``boolean_value`` is False) and the pairing is asserted per row.
        """
        label_df = pl.DataFrame(
            {
                DataSchema.subject_id_name: [2, 1, 2, 1, 2, 1],
                LabelSchema.prediction_time_name: [
                    datetime(2020, 1, 5),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 5),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 5),
                    datetime(2020, 1, 2),
                ],
                "boolean_value": [True, False, False, True, True, False],
                "occurs": [1, 2, 4, 3, 5, 6],
                "query": [11, 22, 44, 33, 55, 66],
                "duration_days": [1.0, 2.0, 4.0, 3.0, 5.0, 6.0],
            }
        )

        result = EveryQueryPytorchDataset.get_task_seq_bounds_and_labels(label_df, _schema_df())

        assert result.height == 6
        for out_row in result.iter_rows(named=True):
            occurs = out_row["occurs"]
            assert (occurs % 2 == 0) is (out_row["boolean_value"] is False), (
                f"annotation row {occurs} paired with the wrong label {out_row['boolean_value']}"
            )
            # `query` and `duration_days` travel with `occurs` on the same source row.
            assert out_row["query"] == occurs * 11
            assert out_row["duration_days"] == float(occurs)

        # No annotation row was dropped or duplicated by the realignment.
        assert sorted(result["occurs"].to_list()) == [1, 2, 3, 4, 5, 6]

    def test_extras_align_when_the_join_engine_reorders_rows(self, monkeypatch):
        """The alignment must not depend on join row order at all (#299).

        Polars' `join` takes `maintain_order=None` by default, so the engine is free to emit
        rows in any order; today's pinned version happens to preserve the left input's, which
        is precisely why an order-dependent implementation passes the tests above. Here every
        join is patched to reverse its output, standing in for a future engine that reorders.
        """
        # Only the lazy layer is patched: `DataFrame.join` delegates to `LazyFrame.join`, so
        # patching both would reverse twice and cancel out. This single hook reverses every
        # join in the call, eager and lazy alike.
        original_join = pl.LazyFrame.join

        def reversing_join(self, *args, **kwargs):
            return original_join(self, *args, **kwargs).reverse()

        monkeypatch.setattr(pl.LazyFrame, "join", reversing_join)

        label_df = pl.DataFrame(
            {
                DataSchema.subject_id_name: [3, 1, 2, 1, 99, 2],
                LabelSchema.prediction_time_name: [
                    datetime(2020, 1, 10),
                    datetime(2020, 1, 3),
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 5),
                ],
                "boolean_value": [True, False, True, True, False, False],
                "occurs": [30, 12, 21, 11, 99, 22],
                "query": [300, 120, 210, 110, 990, 220],
                "duration_days": [3.0, 1.2, 2.1, 1.1, 9.9, 2.2],
            }
        )

        result = EveryQueryPytorchDataset.get_task_seq_bounds_and_labels(label_df, _schema_df())

        assert result.height == 5
        label_lookup = {
            (row[DataSchema.subject_id_name], row[LabelSchema.prediction_time_name]): row
            for row in label_df.iter_rows(named=True)
        }
        for out_row in result.iter_rows(named=True):
            key = (out_row[DataSchema.subject_id_name], out_row[LabelSchema.prediction_time_name])
            expected = label_lookup[key]
            assert out_row["occurs"] == expected["occurs"], f"occurs misaligned for {key}"
            assert out_row["query"] == expected["query"], f"query misaligned for {key}"
            assert out_row["duration_days"] == expected["duration_days"], f"duration misaligned for {key}"

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
