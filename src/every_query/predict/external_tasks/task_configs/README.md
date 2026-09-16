# `external_tasks/task_configs/`

[ACES](https://github.com/justin13601/ACES) task-definition YAMLs: the *inputs* that produce the
task parquets `aces_to_eq.py` converts into EveryQuery's `TaskQuerySchema`. One file per task,
grouped into a directory per task family (`icu/` today).

These are benchmark definitions shared across models: EveryQuery and every baseline it is compared
against are evaluated on labels extracted from these exact configs, so they live in version control
rather than in a scratch directory on whichever cluster ran the extraction.

## Format

Each file is a standalone ACES config (`predicates`, `trigger`, `windows`) preceded by a comment
header giving the clinical question in plain English plus the trigger, prediction time, label, and
cohort restrictions. The header is the spec; the YAML below it is the operationalisation.

`tests/test_aces_task_configs.py` loads every file here with ACES' own parser, so a broken
config fails CI rather than a cluster run. Conventions the configs follow by hand:

- **The file stem is the task name.** `aces_to_eq.yaml` interpolates `task_name` into
    `${ACES_SHARDS_DIR}/${task_name}/held_out`, so extract `icu/mortality_90d.yaml` into
    `$ACES_SHARDS_DIR/icu/mortality_90d/` and the conversion needs no further path wiring.
- **Exactly one labeled window.** ACES emits a single `boolean_value` per task, which
    `aces_to_eq.py` copies verbatim as `task_label`.
- **The prediction time is `input.end`.** `input` is the only window carrying `index_timestamp`,
    and `aces_to_eq.py` joins EQ rows to ACES rows on `(subject_id, prediction_time)`.
- **A minimum of 10 prior events** (`_ANY_EVENT: (10, None)` on `input`), so no task is triggered on
    a subject with essentially no history.

## ICU tasks

All 15 trigger inside an open ICU stay and predict from the triggering event. The ones with a fixed
horizon exclude triggers whose patient leaves the hospital inside that horizon, so a positive label
can't be censored away by discharge; the ones anchored to a downstream event (bounceback,
reintubation, vasopressor reinitiation) instead condition on that event happening within a
qualifying window and then measure from it.

| Task                                     | Clinical question                                                              | Label                   | Horizon             |
| ---------------------------------------- | ------------------------------------------------------------------------------ | ----------------------- | ------------------- |
| `consecutive_low_map`                    | Will the next charted MAP after a MAP < 65 also be < 65?                       | `map_low`               | next MAP, ≤ 6h      |
| `imminent_icu_mortality`                 | Will the patient die in the next 24h?                                          | `death`                 | 24h                 |
| `imminent_hypoglycemia`                  | Given a glucose is drawn in the next 12h, will any reading be < 70 mg/dL?      | `glucose_low`           | 12h                 |
| `new_onset_atrial_fibrillation`          | Will afib be documented in the next 24h, with none earlier in the stay?        | `afib`                  | 24h                 |
| `acute_deterioration_event`              | Will a cardiac arrest, code blue, or rapid response occur in the next 24h?     | `deterioration`         | 24h                 |
| `mortality_90d`                          | Will the patient die within 90 days?                                           | `death`                 | 90d                 |
| `icu_bounceback_or_death`                | Given ICU discharge within 48h, will they return or die within 48h of leaving? | `bounceback_or_death`   | 48h post-discharge  |
| `reintubation_after_extubation`          | Given extubation within 48h, will they be reintubated within 72h of it?        | `intubation`            | 72h post-extubation |
| `prolonged_ventilation_past_day14`       | Will they be extubated before day 14, among patients alive at day 14?          | `extubation`            | day 14 of episode   |
| `extubation_before_tracheostomy`         | Among survivors, will the episode end in extubation rather than tracheostomy?  | `extubation`            | episode end         |
| `prolonged_ventilation_past_day21`       | Will they be extubated before day 21, among patients alive at day 21?          | `extubation`            | day 21 of episode   |
| `discharge_to_facility_ventilated`       | Will the episode end with discharge to a facility while still ventilated?      | `discharge_to_facility` | episode end         |
| `mcs_on_vasopressors`                    | On pressors with no MCS, will MCS be initiated in the next 48h?                | `mcs`                   | 48h                 |
| `second_vasopressor_added`               | On exactly one pressor, will a second be started in the next 6h?               | `pressor_start`         | 6h                  |
| `vasopressor_reinitiation_after_weaning` | Given pressors stopped within 48h, will they restart within 72h of that stop?  | `pressor_start`         | 72h post-stop       |

### Codes these configs assume

Hospital discharge is coded with an explicit disposition suffix,
`HOSPITAL_DISCHARGE//<DISPOSITION>`. Configs that use discharge as a censoring gate
enumerate the live dispositions (`HOME`, `SNF`, `REHAB`, `LTACH`, `AMA`) rather than
matching every discharge, so the gate does not fire on the discharge record that
accompanies a death and delete exactly the rows where the outcome occurred.
`discharge_to_facility_ventilated` matches all dispositions with a regex, because there
any discharge ends the ventilation episode.

The remaining names are the intended canonical ones: `ICU_ADMISSION`, `ICU_DISCHARGE`,
`DEATH`, `INTUBATION`, `EXTUBATION`, `CODE_BLUE`, `RAPID_RESPONSE`, the `MED_START//` and
`MED_END//` pressor codes, and the `PROC//` procedure codes. `LAB//220052` (MAP),
`LAB//50931` / `LAB//50809` / `LAB//225664` (glucose) and the `CHART//220048//` afib
regex are MIMIC itemids. Expect to remap per cohort before extraction.

Note that polars, which ACES uses for predicate matching, compiles regexes with the Rust
regex crate: `\b` works, but look-ahead and look-behind are not supported at all.

## Usage

ACES resolves a task config as `${cohort_dir}/${cohort_name}.yaml`, so point `cohort_dir` at this
directory and `cohort_name` at the path below it, the same string you then pass to `aces_to_eq` as
`task_name`:

```bash
TASK_CONFIGS="$(python -c 'import every_query, pathlib; print(pathlib.Path(every_query.__file__).parent / "predict/external_tasks/task_configs")')"

# 1. Extract labels with ACES (not an EveryQuery dependency, so install it separately).
#    Check `aces-cli --help` for the data and shard arguments your cohort needs.
aces-cli cohort_dir="$TASK_CONFIGS" cohort_name=icu/mortality_90d

# 2. Convert the ACES parquets into per-code TaskQuerySchema parquets.  `duration_days`
#    stamps the task's horizon onto every output row; ACES carries it in the config
#    rather than per row, so it has to be passed explicitly here.
python -m every_query.predict.external_tasks.aces_to_eq \
	task_name=icu/mortality_90d duration_days=90
```
