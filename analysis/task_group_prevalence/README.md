# Gain by task group and prevalence

Two-panel figure of EveryQuery's AUROC gain over the autoregressive baseline on the clinical task
evaluation, split into (a) ICU stay (MIMIC-IV, NWICU) and (b) longitudinal care (AMC).

- Each row is a task group.
- Each dot is the mean delta AUROC (EveryQuery minus autoregressive) over that group's tasks in one
  outcome prevalence bin (<1%, 1-5%, >=5%). Darker dots are rarer outcomes.
- A gradient line joins a group's bins.
- Rows are ordered by the group's mean delta over all of its tasks, largest at the top.

## Files

- `plot_gain_by_prevalence.py` draws the figure.
- `task_groups.csv` assigns each evaluated query a `setting` (`ICU` or `longitudinal`) and a
  `task_group`, keyed by `source` and `query`.
  - `source` is the results file stem without its `_minposN` suffix.
  - `query` is the `Query Specification` string.

## Usage

The results CSVs are not checked in. Pass them as arguments:

```bash
python analysis/task_group_prevalence/plot_gain_by_prevalence.py \
    mimic_icu_24h_minpos40.csv nwicu_icu_24h_minpos40.csv \
    amc_outpatient_visit_general_minpos40.csv amc_outpatient_visit_hf_codes_minpos40.csv \
    amc_outpatient_visit_t2d_codes_minpos40.csv
```

This writes `analysis/figures/eq_gain_by_group_prevalence.{pdf,png}`. Use `--out-dir` to write the
figure somewhere else.

If a results file has a query that `task_groups.csv` doesn't cover, the script stops and lists the
missing rows. Add them to the CSV with a setting and task group.

## How the task groups were assigned

The groups were assigned once, by the target code of each query:

| Target code | Group |
|---|---|
| `MEDS_DEATH` | Mortality |
| `ICU_DISCHARGE` | Length of stay |
| `INFUSION_START` (high-dose bin) | Vasopressor |
| `LAB` or `LOINC` with a value range | Abnormal lab |
| `LOINC` with `//ANY` | Lab monitoring |
| `ICD10CM` | Secondary diagnosis or Complication |
| `RxNorm` | Medication initiation |
| `Visit` | Hospitalization or Return visit |

Code names came from the MIMIC `d_items`/`d_labitems` tables and the OMOP concept vocabulary.

Notes on specific groups:

- **NWICU procedures:** itemids `787541` (mechanical ventilation) and `792843` (non-invasive
  ventilation) come from the NWICU `d_items` table.
- **NWICU labs:** the `LAB//1000xx` analytes are inferred from their units and thresholds. They don't
  affect the grouping, since all of them are Abnormal lab.
- **Response to medication:** these tasks are `Visit/IP` outcomes with an extra `RxNorm` clause.
