# Plan: Stage 4 — Per-shard labeling worker (Issue #209)

## Summary

Implement `label_one_shard()` — a self-contained per-shard labeling function that reads a Stage 3 index partition + shard events, labels via the existing `join_asof` core, and writes the final dataset shard atomically. Also fix the float-duration bug in `evaluate_index_df()` (currently uses `pl.duration(days=...)` which requires integers).

## Files to change

### 1. `src/every_query/generate_tasks/sample_tasks.py`

**A. Fix float-duration in `evaluate_index_df()` (line 630)**

The current code:
```python
duration_expr = pl.duration(days=pl.col(TaskQuerySchema.duration_days_name))
```
Must change to:
```python
duration_expr = pl.duration(seconds=pl.col(TaskQuerySchema.duration_days_name) * 86_400)
```
This is required because `pl.duration(days=...)` truncates to integer days, but the redesigned pipeline uses float `duration_days` (from `QueryDistribution`). The `* 86_400` conversion preserves sub-day precision. Both uses of `duration_expr` (line 631 `window_end` and line 634 `event_in_window`) benefit from the fix.

**B. Add `label_one_shard()` function (new, after the path helpers section ~line 1012)**

```python
def label_one_shard(
    shard: str,
    index_dir: Path,
    data_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> tuple[str, str]:
```

Steps:
1. Compute `final = out_dir / f"{shard}.parquet"` (same as `final_output_path` but takes pre-resolved dirs to avoid passing `split` separately — the caller already partitioned by split).
2. **Skip-on-success**: if `not overwrite and final.exists()`, return `(shard, "skipped")`. Atomic writes (invariant 8) guarantee a present file is complete.
3. **Stale temp cleanup**: glob `out_dir / f".{shard}.parquet.tmp.*"` and unlink each. These are orphans from crashed prior workers.
4. Read `index_df = pl.read_parquet(index_dir / f"{shard}.parquet")` — the Stage 3 index partition. It already carries `prediction_time` (invariant 3: no index-space work in Stage 4).
5. Read `events_df = _read_event_shard(data_dir / f"{shard}.parquet")` — reuses the existing helper that normalizes code→Utf8, time→Datetime("us").
6. `max_time = compute_max_time_per_subject(events_df)` — one groupby for censoring.
7. `labeled = evaluate_index_df(index_df, events_df, max_time)` — the existing labeling core (now with the float-duration fix from step A).
8. `aligned = TaskQuerySchema.align(labeled.to_arrow())` — coerce to canonical schema at the write boundary.
9. `_atomic_write_parquet(pl.from_arrow(aligned), final)` — atomic sibling-temp write.
10. Return `(shard, "labeled")`.

**C. Add `_clean_stale_temps()` helper (small utility)**

```python
def _clean_stale_temps(out_dir: Path, shard: str) -> int:
```

Globs `out_dir / f".{shard}.parquet.tmp.*"` and unlinks each. Returns count removed. Called at the top of `label_one_shard` before any writes. This prevents temp file accumulation from crashed workers.

### 2. `tests/test_sample_tasks.py`

Add a `TestLabelOneShard` test class covering:

**a. `test_basic_labeling`** — Create a minimal on-disk fixture (index partition + events shard as parquets in tmp dirs), call `label_one_shard()`, verify:
- Returns `(shard, "labeled")`
- Output file exists at `out_dir/{shard}.parquet`
- Output has correct columns: `subject_id, prediction_time, query, duration_days, boolean_value`
- Row count matches index partition
- Labels are correct (True/False/null) against known ground truth

**b. `test_skip_on_success`** — Write a dummy output file, call with `overwrite=False`, verify returns `(shard, "skipped")` and file is unchanged.

**c. `test_overwrite`** — Write a dummy output file, call with `overwrite=True`, verify returns `(shard, "labeled")` and file is replaced with fresh labels.

**d. `test_float_duration_labeling`** — Use a non-integer `duration_days` (e.g., 1.5) to verify the float-duration fix works end-to-end. Create events at times that distinguish integer-day truncation from correct float-duration behavior.

**e. `test_censoring_logic`** — Verify the three-valued label semantics:
- Event in window → `True`
- No event, window fully observed → `False`  
- No event, window extends past max_time → `null`

**f. `test_stale_temp_cleanup`** — Create orphan `.{shard}.parquet.tmp.*` files in `out_dir`, call `label_one_shard()`, verify they are cleaned up.

**g. `test_empty_index_partition`** — Pass an empty index partition (0 rows, correct schema), verify output is a valid 0-row parquet with the right schema.

## What this plan does NOT include (deferred to other issues)

- **Orchestration** (#210): `ProcessPoolExecutor` fan-out, `resolve_workers()`, `POLARS_MAX_THREADS` pinning, the new `main()` driver. Stage 4 is the worker function; the pool that calls it is #210's scope.
- **Stage 3** (#208): Writing the `_index/{shard}.parquet` partitions. This plan assumes Stage 3 output exists on disk.
- **Post-Stage-4 row-count assertion** (#210): "assert union row count == `num_queries * num_contexts_per_query`" belongs in the orchestrator.
- **Determinism tests** (#211): Cross-process reproducibility validation.

## Implementation order

1. Fix float duration in `evaluate_index_df()` (one-line change)
2. Add `_clean_stale_temps()` helper
3. Add `label_one_shard()` function
4. Add tests
5. Run `uv run pytest tests/test_sample_tasks.py -v` to verify all existing + new tests pass
6. Run `uv run ruff check src/every_query/generate_tasks/sample_tasks.py tests/test_sample_tasks.py` for lint
