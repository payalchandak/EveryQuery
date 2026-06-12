# EveryQuery Task Sampler Design Doc

## Pipeline Inputs and Outputs
Inputs:
- Query Distribution: (e.g. uniform vs weighted)
- MEDS Dataset (D)
- Number of queries to sample (N)
- Number of patient context samples per sampled query (M)

Output:
- TaskQuerySchema parquet of len (N*M)
## Stages
1) Sample `N` Queries
2) Sample `N*M` patient contexts
3) Match up each query with a patient context
4) Compute Censor and Occurs label


## Query Sampling Stage Details
### Function Signature
```python
def sample_queries(
    distribution:QueryDistribution, 
    N:int) --> list[QuerySpec]
```
### Datstructures
```python
@dataclass(frozen=True)
class QuerySpec:
    """A single pre-training query/task: one query code and one prediction-window duration (in days)."""
    code: str
    duration_days: float
@dataclass(frozen=True)
class QueryDistribution:
    """Defines a distribution that that will provide samples of queries for EQ training. A sample from the distribution is a single QuerySpec i.e. a code and duration tuple"""
    codes: list[str]
    max_duration_days: float
```
    
## Patient Context Sampling Stage Details
### Function Signature
```python
def sample_patient_contexts(
    dataset: MEDSDataset,
    distribution: ContextDistribution,
    N: int,
    M: int) --> list[PatientContext]
```
`N` is the number of sampled queries and `M` is the number of patient context samples per sampled query, so this stage draws `N*M` contexts in total (one flat list, ordered so the first `M` belong to query 0, the next `M` to query 1, and so on). Rather than re-running the sampler `N` times, all `N*M` contexts are drawn in a single seeded call and then sliced into per-query chunks of size `M` downstream (see Matching Stage) — under iid sampling this is equivalent to `M` independent draws per query but requires only one seed.

### Datastructures
```python
@dataclass(frozen=True)
class PatientContext:
    """A single patient context: one subject and the prediction time that cuts off their observable history. Everything strictly before `prediction_time` is model input; the label is evaluated over the window starting at `prediction_time`."""
    subject_id: int
    prediction_time: datetime

@dataclass(frozen=True)
class ContextDistribution:
    """Defines a distribution that provides samples of patient contexts for EQ training. A sample from the distribution is a single PatientContext i.e. a (subject_id, prediction_time) tuple drawn from the eligible (subject, event-time) pairs in the dataset."""
    min_context_events: int   # a (subject, time) pair is eligible only once the subject has >= this many prior events
    with_replacement: bool    # True for PT, so N*M may exceed the number of eligible pairs
    duration_days_distribution: uniform | log-uniform
```

### Sampling Semantics
- **Candidate pool.** The candidate prediction times are every event time at which a subject has already accumulated at least `min_context_events` prior events. 
- **Uniformity.** Each draw is uniform over the deduplicated set of eligible `(subject_id, prediction_time)` pairs. Sampling at the event-pair level (rather than first sampling a subject, then a time) means subjects with longer histories are proportionally more likely to be drawn — contexts are uniform over *eligible time points*, not over patients.
- **Replacement.** Sampling is with replacement so the caller can request `N*M` larger than the number of eligible pairs in a shard; for pre-training, iid-ness of contexts matters more than strict coverage of distinct time points.

### Determinism
The context draw is seeded from `(seed, input_shard)` via `derive_seed` (no task-shard axis in this design — queries and contexts are both drawn once per shard from the same `seed`, split only by a stage tag so the two draws don't correlate). The seed varies across the patient axis (`input_shard`) while remaining reproducible for a fixed `(seed, input_shard)`. Because polars' `.unique()` is hash-ordered and order-unstable across calls, the eligible-pair pool is explicitly sorted by `(subject_id, prediction_time)` before sampling so that a position-based sampler is reproducible.

### Edge Cases
- `N*M == 0` or an empty eligible pool returns an empty list (zero-row frame), not an error.
- A subject whose entire record is shorter than `min_context_events` contributes no candidates.



## Match up Query and Patient Context Stage Details
### Function Signature
```python
def match_queries_and_contexts(
    queries: list[QuerySpec],     # len N
    contexts: list[PatientContext]  # len N*M
) --> pl.DataFrame  # len N*M, one row per (query, context) pair
```

Zip the `N` queries with the `N*M` contexts by assigning query `i` to its block of `M` consecutive contexts (`contexts[i*M : (i+1)*M]`). Each output row carries the query's `code` and `duration_days` alongside the context's `subject_id` and `prediction_time` — i.e. the unlabeled `(query, context)` pairs that the next stage labels.

Requires `len(contexts) == len(queries) * M`. Under iid sampling this block assignment is equivalent to drawing `M` fresh contexts per query.

## Compute Censor and Occurs label Stage Details
### Function Signature
```python
def compute_labels(
    pairs: pl.DataFrame,    # output of the matching stage, len N*M
    dataset: MEDSDataset,
) --> pl.DataFrame  # TaskQuerySchema, len N*M
```

For each `(query, context)` row, look at the window `(prediction_time, prediction_time + duration_days]` for that subject:

- **censored** — `prediction_time + duration_days > max_time[subject_id]`, i.e. the subject's record ends before the window closes, so we never observe whether the event fires.
- **occurs** — not censored, and an event with the query `code` falls strictly within the window.

These collapse into a single nullable `boolean_value` (the `TaskQuerySchema` label): `null` when censored, else `True`/`False` from occurs.

| censored | occurs | boolean_value |
|----------|--------|---------------|
| yes      | —      | null          |
| no       | yes    | True          |
| no       | no     | False         |

`max_time` per subject is computed once per shard (one groupby) so the per-row censor check is an O(1) lookup. Occurrence is resolved for all rows in a single `join_asof(strategy="forward", by=["subject_id", "code"])` against the events table.