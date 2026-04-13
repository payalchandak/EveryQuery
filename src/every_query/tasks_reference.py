"""Reference implementations of the pre-refactor tasks.py functions.

These are verbatim copies of compute_censor_dataframe and build_task_label_matrix
from the main branch before the new-speedup refactor, with one fix: the original
code used `not pl.col("censored")` which raises in current polars. This is corrected
to `pl.col("censored").not_()` — the logic is identical, only the API call changes.

These exist solely for regression testing: tests compare the output of the new
precompute_min_deltas_wide + build_task_for_duration pipeline against these originals.
Do not import these in production code.
"""

import polars as pl
from tqdm import tqdm


def compute_censor_dataframe(
    events_df: pl.DataFrame,
    min_context_per_subject: int,
    duration: dict[str, int],
) -> pl.DataFrame:
    """Compute per-subject prediction times and whether they are censored.

    Censoring is defined as having less than <duration> of future data after prediction_time. Retain at least
    min_context_per_subject context tokens (not times) before the first prediction_time.
    """
    return (
        events_df.with_columns(pl.col("time").cum_count().over("subject_id").alias("context_cumsum"))
        .filter(pl.col("context_cumsum") >= min_context_per_subject)
        .select(["subject_id", "time"])  # candidate prediction times
        .unique()
        .rename({"time": "prediction_time"})
        .join(
            events_df.group_by(["subject_id"]).agg(pl.col("time").last().alias("record_end_time")),
            on="subject_id",
            how="left",
        )
        .with_columns((pl.col("record_end_time") - pl.col("prediction_time")).alias("future_duration"))
        .with_columns((pl.col("future_duration") < pl.duration(**duration)).alias("censored"))
        .select(["subject_id", "prediction_time", "censored"])
    )


def build_task_label_matrix(
    events_df: pl.DataFrame,
    censor_df: pl.DataFrame,
    query_codes: list[str],
    duration: dict[str, int],
) -> pl.DataFrame:
    """Create a wide task label matrix.

    - For censored rows: label columns for each query are null (Boolean).
    - For uncensored rows: label is True if the query event occurs within
      (prediction_time, prediction_time + duration), else False.

    Note: the original source used `not pl.col("censored")` which raises in current
    polars. This reference uses `pl.col("censored").not_()` — same logic.
    """
    censor_true = censor_df.filter(pl.col("censored"))
    censor_true_wide = censor_true.with_columns(
        [pl.lit(None).alias(query).cast(pl.Boolean) for query in query_codes]
    )

    censor_false = censor_df.filter(pl.col("censored").not_())
    censor_false_time = censor_false.drop("censored").with_row_index()
    censor_false_index = censor_false_time.select("index")

    pieces: list[pl.DataFrame] = [censor_false]
    for query in tqdm(query_codes):
        pieces.append(
            censor_false_time.join(
                events_df.filter(pl.col("code") == query).drop("code").rename({"time": f"{query}_time"}),
                on="subject_id",
                how="left",
            )
            .filter(
                (pl.col(f"{query}_time") > pl.col("prediction_time"))
                & (pl.col(f"{query}_time") < (pl.col("prediction_time") + pl.duration(**duration)))
            )
            .select(["index"])
            .unique()
            .with_columns(pl.lit(True).alias(query))
            .join(censor_false_index, on="index", how="right")
            .with_columns(pl.col(query).fill_null(False))
            .select([query])
        )

    censor_false_wide = pl.concat(pieces, how="horizontal")
    assert sum(censor_false_wide.null_count()).item() == 0  # Ensure no label nulls remain on uncensored rows

    return pl.concat([censor_true_wide, censor_false_wide], how="vertical")
