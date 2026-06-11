# Design Doc: Pre-training Task Generation

## 1. Goal

Generate pre-training tasks for EQ by sampling queries, sampling patient
contexts for each query, and computing answers — producing a task dataframe for
training.

This is a from-scratch design built around the following spec.

---

## 2. Inputs
**Note:** A "patient context" is a (subejct_id,prediction_time)

| Input | Description | Example |
|---|---|---|
| **Query distribution** | Distribution over the query space to sample queries from. | How are codes drawn (uniform vs weighted)? How are duration days sampled? (Log uniform vs uniform)? |
| **Patient context \| query distribution** | Figure out | Figure out | Whole dataset (not a single shard). Access pattern / working-set bound? |
| **Number of query samples** (`N`) | How many queries to draw. | |
| **Number of patient-context samples per query** (`M`, default **1**) | Contexts drawn per sampled query. | `M=1` maximizes query diversity for a fixed training budget — **test this** (§5). |

**Things to clarify**
- Why is the patient context sampled conditioned on the query? We didn't do this before but I'm assuming this is for extensibility?
---

## 3. Algorithm

1. **Sample queries.** Randomly sample a list of `N` queries from the query
   distribution.
2. **Sample contexts.** For each sampled query, sample `M` patient contexts
3. **Compute answers.** For each `(query, patient context)` pair, compute the
   `(can_answer, answer | can_answer)` pair.
4. **Compile.** Assemble all results into the task dataframe.

```
query_dist ──sample N──▶ [queries]
                            │
              for each query, sample M contexts
                            ▼
                 [(query, patient_context)] of len (NxM)
                            │
            compute (can_answer, answer | can_answer)
                            ▼
                    task dataframe
```

**To specify per step:**
- Step 1 — sampling with/without replacement? 
- Step 2 — what does "sample contexts for a query" draw from; with replacement?
  what if no valid context exists for a query (is this possible)?
- Step 3 — `answer` is defined only when `can_answer` is true; what fills
  `answer` otherwise? What is the parallelization unit (this is the expensive
  step)?
- Step 4 — output schema (columns, dtypes).

---

## 4. Output

Task dataframe. Define the schema:

| Column | Type | Notes |
|---|---|---|
| query (repr / id) | | |
| subject_id | | |
| prediction_time | | |
| can_answer | bool | |
| answer | nullable | valid iff `can_answer` |

---

## 5. Key Design Question: contexts per query (`M`)

`M = 1` is the default because **one context per query maximizes query diversity
for a fixed training budget** — but this is a hypothesis to test, not an
assumption.

- Make `M` a swept knob.
- Experiment: vary `M ∈ {1, ...}` at fixed total budget (`N × M` fixed),
  measure downstream pre-training quality.
- Decision criterion: _fill in_ (which metric, what counts as a win).

---

## 6. Correctness & Reproducibility

- **Determinism:** generation is a pure function of its inputs + seed; same seed
  → identical task dataframe.
- **Seeding:** decide how randomness splits across the query axis vs. the
  context axis, and across parallel workers, so results don't depend on worker
  count or scheduling. _(Specify the seed-derivation scheme here.)_
- **Tests:**
  - same seed → identical output (regression);
  - distinct seeds → distinct draws;
  - sampled query frequencies match the query distribution within tolerance;
  - `answer` is null iff `can_answer` is false.

---

## 7. Open Questions

- [ ] Exact query space + query distribution parameterization.
- [ ] Patient-context-given-query conditional: validity + query-dependence.
- [ ] `M` experiment (§5).
- [ ] Behavior when a query has no valid context, or `can_answer` is frequently false.
- [ ] Parallelization granularity for answer computation over the whole dataset.
- [ ] Output schema finalization.
