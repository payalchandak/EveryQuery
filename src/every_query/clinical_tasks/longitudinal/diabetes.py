"""Longitudinal diabetes tasks, anchored at an outpatient visit among patients with type
2 diabetes.

**Columbia NYP, not MIMIC.**  See ``outpatient_visit.py`` for why the two datasets share
no codes.

The type 2 diabetes restriction is carried by the anchor spec, not by the queries: no
sub-question here narrows the denominator to diabetics.  Confirmed, not assumed.

Lab thresholds follow the rule used in ``icu/labs.py``: values are binned into population
deciles, so each task takes the outermost bin and states that bin's real boundary, kept
only where the resulting question stands on its own.

- HbA1c above 9.0 -> ``>= 8.9``, poor glycemic control against a target below 7.
- creatinine above 2.0 -> ``>= 2.34``, significant renal dysfunction.
- eGFR below 30 -> ``< 45``, CKD stage 3b or worse.

"Is it measured" tasks use the unbinned code, which Columbia carries alongside the bins.
The urine albumin-to-creatinine ratio takes ``LOINC/14959-1``
(microalbumin/creatinine), the diabetic screening assay, in preference to
``LOINC/9318-7``; the latter appears in the vocabulary with no value bins at all, which
suggests it carries little volume.

The metformin task conditions on ``RxNorm/6809``, the metformin ingredient, which
Columbia carries as a single code.  Its conditioning guard uses ``forced_answer: True``
and a 30-day window of its own, narrower than the task's 1-year horizon.

Most of this specification is not here.  Seventeen tasks need derived code sets and are
deferred: diabetic ketoacidosis (2 codes), CKD stage 3 (4), diabetic retinopathy (6),
diabetic foot ulcer (11), heart failure (17), genitourinary infection (14), lower
extremity amputation, retinal examination, and the insulin, GLP-1, SGLT2, statin and
metformin concepts, all of which span many RxNorm codes.  Glucose below 54 was dropped:
its outermost bin is ``< 77``, which is inside the normal range of 70 to 100 and so is
not hypoglycemia.
"""

TASKS = {
    "At an outpatient visit, among patients with type 2 diabetes, does an HbA1c at or above 8.9 appear within 1 year, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
            "outcome": "HbA1c",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": False,
            },
            {
                "query": "LOINC/4548-4//UCUM/%//value_[8.9,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, does a creatinine at or above 2.34 appear within 2 years?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "2y",
            "outcome": "creatinine",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 730,
                "forced_answer": False,
            },
            {
                "query": "LOINC/2160-0//UCUM/mg/dL//value_[2.34,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 730,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, does death occur within 5 years?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "5y",
            "outcome": "death",
        },
        "query": [
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1825,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, is an HbA1c measured within 6 months, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "6mo",
            "outcome": "HbA1c measurement",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 180,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 180,
                "forced_answer": False,
            },
            {
                "query": "LOINC/4548-4",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 180,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, is a urine albumin to creatinine ratio measured within 1 year, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
            "outcome": "urine ACR measurement",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": False,
            },
            {
                "query": "LOINC/14959-1",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, given that metformin is prescribed within 30 days, does an eGFR below 45 appear within 1 year, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
            "outcome": "eGFR",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": False,
            },
            {
                "query": "RxNorm/6809",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": True,
            },
            {
                "query": "LOINC/98979-8//UCUM/mL/min/1.73.m2//value_[-inf,45.0)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
}
