import os

import polars as pl
from dotenv import load_dotenv
from meds import held_out_split, train_split, tuning_split
from tqdm import tqdm

load_dotenv()


def read_event_shard(file_path: str) -> pl.DataFrame:
    """Read a single shard parquet file and return processed events.

    Returns DataFrame with columns: subject_id, time, code. Rows are unique and sorted by subject_id, time.
    """
    return (
        pl.read_parquet(file_path)
        .select(["subject_id", "time", "code"])
        .unique()
        .sort(["subject_id", "time"])
    )


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


def read_query_codes(read_dir: str) -> list[str]:
    """Read the universe of possible query codes from metadata/codes.parquet."""
    codes_df = pl.read_parquet(f"{read_dir}/metadata/codes.parquet")
    return codes_df.select("code").unique().to_series().to_list()


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
    """
    censor_true = censor_df.filter(pl.col("censored"))
    censor_true_wide = censor_true.with_columns(
        [pl.lit(None).alias(query).cast(pl.Boolean) for query in query_codes]
    )

    censor_false = censor_df.filter(not pl.col("censored"))
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


if __name__ == "__main__":
    read_codes_dir = os.environ.get("PROCESSED")
    read_dir = os.environ.get("INTERMEDIATE")
    write_dir = f"{os.environ.get('TASK_DIR')}/all"
    min_context_per_subject = 50
    duration = {"minutes": 0, "hours": 0, "days": 30, "weeks": 0}

    for split in [train_split, tuning_split, held_out_split]:
        shard_directory = f"{read_dir}/data/{split}"
        write_directory = f"{write_dir}/{split}"
        os.makedirs(write_directory, exist_ok=True)
        for file_name in os.listdir(shard_directory):
            f = f"{write_directory}/{file_name}"
            if not file_name.endswith(".parquet"):
                continue
            if os.path.exists(f):
                print(f"Skipping {f}. Already exists.")
                continue
            events_df = read_event_shard(f"{shard_directory}/{file_name}")
            print("Completed read_event_shard")
            censor_df = compute_censor_dataframe(events_df, min_context_per_subject, duration)
            print("Completed compute_censor_dataframe")
            query_codes = read_query_codes(read_codes_dir)
            print("Completed read_query_codes")
            task_df = build_task_label_matrix(events_df, censor_df, query_codes, duration)
            print("Completed build_task_label_matrix")
            task_df.write_parquet(f)
