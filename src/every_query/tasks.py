import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor

import hydra
import numpy as np
import polars as pl
from meds import held_out_split, train_split, tuning_split
from omegaconf import DictConfig
from tqdm import tqdm

from every_query._env import ensure_env


def read_event_shard(file_path: str) -> pl.DataFrame:
    """Read a single shard parquet file and return processed events.

    Returns DataFrame with columns: subject_id, time, code. Rows are unique and sorted by subject_id, time.
    The ``code`` column is cast to Categorical so equality filters inside the worker pool run on the
    dictionary-encoded path rather than Utf8 string comparison.
    """
    return (
        pl.read_parquet(file_path)
        .select(["subject_id", "time", "code"])
        .unique()
        .sort(["subject_id", "time"])
        .with_columns(pl.col("code").cast(pl.Categorical))
    )


def read_query_codes(read_dir: str) -> list[str]:
    """Read the universe of possible query codes from metadata/codes.parquet."""
    codes_df = pl.read_parquet(f"{read_dir}/metadata/codes.parquet")
    return codes_df.select("code").unique().to_series().to_list()


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
            events_df.group_by(["subject_id"]).agg(pl.col("time").max().alias("record_end_time")),
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

    Uses join_asof(strategy="forward") per code to avoid the O(n_pred * n_events) cross-join
    that caused OOM on large shards. The asof work is chunked in groups of 500 codes with a
    bounded thread pool so the horizontal concat intermediate never fragments into a wide,
    memory-thrashing frame.

    Events are bucketed once up front via ``partition_by("code")`` so each worker does a dict
    lookup instead of a full ``filter`` pass over the whole shard — this turns the per-code
    cost from O(n_events) into O(n_events_for_that_code).
    """
    pred_sorted = base_df.sort(["subject_id", "prediction_time"])
    if not query_codes:
        return pred_sorted

    # Shift by 1µs so join_asof(strategy="forward") gives time >= prediction_time+1µs,
    # which is equivalent to time > prediction_time for µs-precision datetimes.
    pred_keys = pred_sorted.select(
        "subject_id",
        (pl.col("prediction_time") + pl.duration(microseconds=1)).alias("_pt_shifted"),
        "prediction_time",
    ).set_sorted("subject_id")

    # Single pass: bucket events by code so per-code work is a dict lookup, not a full scan.
    # Sorting by (subject_id, time) once here means every partition slice is already sorted
    # for join_asof, so we can skip the per-code .sort(...) and set_sorted on "subject_id".
    raw_partitions = (
        events_df.select(["subject_id", "time", "code"])
        .sort(["subject_id", "time"])
        .partition_by("code", as_dict=True)
    )
    code_to_events: dict[str, pl.DataFrame] = {}
    for key, df in raw_partitions.items():
        code_str = key[0] if isinstance(key, tuple) else key
        code_to_events[code_str] = df.drop("code").set_sorted("subject_id")
    del raw_partitions

    # Reusable all-null delta column for codes that do not appear in this shard.
    null_delta_series = pl.repeat(None, n=pred_sorted.height, dtype=pl.Duration("us"), eager=True)

    def _compute_code_delta(code: str) -> pl.DataFrame:
        code_events = code_to_events.get(code)
        if code_events is None:
            return null_delta_series.alias(code).to_frame()
        asof = pred_keys.join_asof(
            code_events,
            by="subject_id",
            left_on="_pt_shifted",
            right_on="time",
            strategy="forward",
        )
        delta_col = (
            pl.when(pl.col("time").is_not_null())
            .then(pl.col("time") - pl.col("prediction_time"))
            .otherwise(pl.lit(None).cast(pl.Duration("us")))
            .alias(code)
        )
        return asof.select(delta_col)

    # Respect the NUMA thread budget Polars was pinned to; hard-cap at 4 regardless.
    max_workers = min(4, int(os.environ.get("POLARS_MAX_THREADS", "4")))
    chunk_size = 500
    result = pred_sorted
    for i in range(0, len(query_codes), chunk_size):
        chunk = query_codes[i : i + chunk_size]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            chunk_cols = list(pool.map(_compute_code_delta, chunk))
        result = pl.concat([result, *chunk_cols], how="horizontal")
        del chunk_cols
        gc.collect()

    return result


def build_task_for_duration(
    min_deltas_wide: pl.DataFrame,
    query_codes: list[str],
    duration: dict[str, int],
) -> pl.DataFrame:
    """Build the task label matrix for a specific duration from precomputed min deltas.

    Pure column arithmetic — no joins. Returns the same schema as tasks_reference.build_task_label_matrix.
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

    Returns a sorted list of exactly ``n`` unique integer durations.  Uses an
    oversampling/retry loop so that deduplication after rounding never yields
    fewer than ``n`` values.  If fewer than ``n`` distinct integers exist in
    [low, high], returns all of them.
    """
    if low < 1:
        raise ValueError(f"low must be >= 1 (got {low}); log-uniform requires positive bounds")
    max_possible = high - low + 1
    n = min(n, max_possible)
    rng = np.random.default_rng(seed)
    log_low, log_high = np.log(low), np.log(high)
    unique: set[int] = set()
    batch_size = n
    while len(unique) < n:
        raw = np.exp(rng.uniform(log_low, log_high, size=batch_size))
        unique.update(round(x) for x in raw)
        batch_size = max(n - len(unique), 1) * 2
    return sorted(unique)[:n]


@hydra.main(version_base=None, config_path=".", config_name="tasks_config")
def main(cfg: DictConfig) -> None:
    ensure_env()
    shard_index = int(cfg.shard_index) if cfg.shard_index is not None else None

    read_codes_dir = os.environ["PROCESSED"]
    read_dir = os.environ["INTERMEDIATE"]
    task_dir = os.environ["TASK_DIR"]

    if cfg.durations is not None:
        durations = sorted(cfg.durations)
    else:
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

    query_codes = read_query_codes(read_codes_dir)
    print("Completed read_query_codes")

    for split, file_name in shards_to_process:
        shard_directory = f"{read_dir}/data/{split}"
        # Keep the categorical dictionary stable across worker threads for this shard.
        with pl.StringCache():
            events_df = read_event_shard(f"{shard_directory}/{file_name}")
            print(f"Completed read_event_shard for {split}/{file_name}")

            # One-time precomputation per shard
            base_df = compute_base_prediction_times(events_df, cfg.min_context)
            print("Completed compute_base_prediction_times")
            min_deltas = precompute_min_deltas_wide(events_df, base_df, query_codes)
            print("Completed precompute_min_deltas_wide")

            # events_df and base_df are no longer needed — free before the duration loop
            del events_df, base_df
            gc.collect()

            # Fast per-duration loop
            for days in tqdm(durations, desc=f"{split}/{file_name}"):
                write_directory = f"{task_dir}/{days}/{split}"
                out_path = f"{write_directory}/{file_name}"
                if os.path.exists(out_path):
                    continue
                os.makedirs(write_directory, exist_ok=True)
                task_df = build_task_for_duration(min_deltas, query_codes, {"days": days})
                task_df.write_parquet(out_path)

            # Free the wide min_deltas before loading the next shard
            del min_deltas
            gc.collect()


if __name__ == "__main__":
    main()
