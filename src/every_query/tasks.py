import json
import os

import hydra
import numpy as np
import polars as pl
from dotenv import load_dotenv
from meds import held_out_split, train_split, tuning_split
from omegaconf import DictConfig
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


# ---------------------------------------------------------------------------
# Optimized functions
# ---------------------------------------------------------------------------


def compute_base_prediction_times(
    events_df: pl.DataFrame,
    min_context_per_subject: int,
) -> pl.DataFrame:
    """Compute prediction times and future_duration once per shard (duration-independent).

    Returns DataFrame with columns: subject_id, prediction_time, future_duration.
    """
    return (
        events_df.with_columns(pl.col("time").cum_count().over("subject_id").alias("context_cumsum"))
        .filter(pl.col("context_cumsum") >= min_context_per_subject)
        .select(["subject_id", "time"])
        .unique()
        .rename({"time": "prediction_time"})
        .join(
            events_df.group_by(["subject_id"]).agg(pl.col("time").last().alias("record_end_time")),
            on="subject_id",
            how="left",
        )
        .with_columns((pl.col("record_end_time") - pl.col("prediction_time")).alias("future_duration"))
        .drop("record_end_time")
    )


def derive_censor_for_duration(
    base_df: pl.DataFrame,
    duration: dict[str, int],
) -> pl.DataFrame:
    """Derive the censored column for a specific duration from precomputed base prediction times."""
    return base_df.with_columns(
        (pl.col("future_duration") < pl.duration(**duration)).alias("censored")
    ).select(["subject_id", "prediction_time", "censored"])


def precompute_min_deltas_wide(
    events_df: pl.DataFrame,
    base_df: pl.DataFrame,
    query_codes: list[str],
) -> pl.DataFrame:
    """Precompute the minimum positive time delta to each query code for every prediction time.

    Returns a wide DataFrame with columns:
        subject_id, prediction_time, future_duration, <code_1>, <code_2>, ...
    where each code column contains the minimum duration until that code's next occurrence
    strictly after prediction_time (or null if it never occurs).

    Uses join_asof(strategy="forward") per code to avoid the O(n_pred × n_events) cross-join
    that caused OOM on large shards.
    """
    pred_sorted = base_df.sort(["subject_id", "prediction_time"])
    # Shift by 1µs so join_asof(strategy="forward") gives time >= prediction_time+1µs,
    # which is equivalent to time > prediction_time for µs-precision datetimes.
    pred_keys = pred_sorted.select(
        "subject_id",
        (pl.col("prediction_time") + pl.duration(microseconds=1)).alias("_pt_shifted"),
        "prediction_time",
    )

    code_cols: list[pl.DataFrame] = []
    for code in query_codes:
        code_events = (
            events_df.filter(pl.col("code") == code)
            .select(["subject_id", "time"])
            .sort(["subject_id", "time"])
        )
        asof = pred_keys.join_asof(
            code_events,
            by="subject_id",
            left_on="_pt_shifted",
            right_on="time",
            strategy="forward",
        )
        # time is the first code event strictly after prediction_time (or null if none)
        delta_col = (
            pl.when(pl.col("time").is_not_null())
            .then(pl.col("time") - pl.col("prediction_time"))
            .otherwise(pl.lit(None).cast(pl.Duration("us")))
            .alias(code)
        )
        code_cols.append(asof.select(delta_col))

    return pl.concat([pred_sorted, *code_cols], how="horizontal")


def build_task_for_duration(
    min_deltas_wide: pl.DataFrame,
    query_codes: list[str],
    duration: dict[str, int],
) -> pl.DataFrame:
    """Build the task label matrix for a specific duration from precomputed min deltas.

    Pure column arithmetic — no joins. Returns the same schema as build_task_label_matrix.
    """
    dur = pl.duration(**duration)

    censored_col = (pl.col("future_duration") < dur).alias("censored")

    # For uncensored rows: code occurred if min_delta < duration; null delta -> False
    # For censored rows: all code columns are null
    code_cols = [
        pl.when(pl.col("future_duration") < dur)
        .then(pl.lit(None).cast(pl.Boolean))
        .otherwise((pl.col(code) < dur).fill_null(False))
        .alias(code)
        for code in query_codes
    ]

    return min_deltas_wide.select(
        "subject_id",
        "prediction_time",
        censored_col,
        *code_cols,
    )


def sample_durations(n: int, low: int, high: int, seed: int) -> list[int]:
    """Sample n durations from a Log-Uniform distribution over [low, high].

    Returns a sorted list of unique integer durations.
    """
    rng = np.random.default_rng(seed)
    log_low, log_high = np.log(low), np.log(high)
    raw = np.exp(rng.uniform(log_low, log_high, size=n))
    durations = sorted({round(x) for x in raw})
    return durations


@hydra.main(version_base=None, config_path=".", config_name="tasks_config")
def main(cfg: DictConfig) -> None:
    shard_index = int(cfg.shard_index) if cfg.shard_index is not None else None

    read_codes_dir = os.environ["PROCESSED"]
    read_dir = os.environ["INTERMEDIATE"]
    task_dir = os.environ["TASK_DIR"]

    durations = sample_durations(cfg.n_durations, 1, 731, cfg.duration_seed)

    # Build deterministic flat list of (split, file_name) pairs
    all_shards: list[tuple[str, str]] = []
    for split in [train_split, tuning_split, held_out_split]:
        shard_directory = f"{read_dir}/data/{split}"
        for file_name in sorted(os.listdir(shard_directory)):
            if file_name.endswith(".parquet"):
                all_shards.append((split, file_name))

    # Write sampled durations (only from index 0 or when running all shards)
    if shard_index is None or shard_index == 0:
        os.makedirs(task_dir, exist_ok=True)
        with open(f"{task_dir}/sampled_durations.json", "w") as f:
            json.dump(durations, f)

    shards_to_process = [all_shards[shard_index]] if shard_index is not None else all_shards

    for split, file_name in shards_to_process:
        shard_directory = f"{read_dir}/data/{split}"
        events_df = read_event_shard(f"{shard_directory}/{file_name}")
        print(f"Completed read_event_shard for {split}/{file_name}")
        query_codes = read_query_codes(read_codes_dir)
        print("Completed read_query_codes")

        # One-time precomputation per shard
        base_df = compute_base_prediction_times(events_df, cfg.min_context)
        print("Completed compute_base_prediction_times")
        min_deltas = precompute_min_deltas_wide(events_df, base_df, query_codes)
        print("Completed precompute_min_deltas_wide")

        # Fast per-duration loop
        for days in tqdm(durations, desc=f"{split}/{file_name}"):
            write_directory = f"{task_dir}/{days}/{split}"
            out_path = f"{write_directory}/{file_name}"
            if os.path.exists(out_path):
                continue
            os.makedirs(write_directory, exist_ok=True)
            task_df = build_task_for_duration(min_deltas, query_codes, {"days": days})
            task_df.write_parquet(out_path)


if __name__ == "__main__":
    main()
