"""Longitudinal diabetes tasks, anchored at an outpatient visit among patients with type
2 diabetes.

**Columbia NYP, not MIMIC.**  See ``outpatient_visit.py`` for why the two datasets share
no codes.

The type 2 diabetes restriction is carried by the anchor spec, not by the queries: no
sub-question here narrows the denominator to diabetics.

.. warning::

    **17 of these 23 tasks are placeholders and cannot be run.**  Their targets or
    conditioning guards keep the specification's concept names (``DIAGNOSIS//...``,
    ``MEDICATION//...``, ``PROCEDURE//...``, ``LAB//GLUCOSE//...``) because the concepts
    span many Columbia codes and the sets have not been derived yet.  Every such task
    carries ``"placeholder": True`` in its metadata, so they can be filtered out until
    the sets land.  The six without that flag resolve fully and are runnable.

Resolved lab thresholds follow the rule from ``icu/labs.py``: values are binned into
population deciles, so a task takes the outermost bin and states that bin's real
boundary.

- HbA1c above 9.0 -> ``>= 8.9``, poor glycemic control against a target below 7.
- creatinine above 2.0 -> ``>= 2.34``, significant renal dysfunction.
- eGFR below 30 -> ``< 45``, CKD stage 3b or worse.

Glucose below 54 is left as a placeholder rather than moved.  Its outermost Columbia bin
is below 77, which sits inside the normal range of 70 to 100, so unlike the three above
it cannot be restated at a bin edge and still mean hypoglycemia.  It needs a custom bin.

"Is it measured" tasks use the unbinned code, which Columbia carries alongside the bins.
Urine ACR takes ``LOINC/14959-1`` (microalbumin/creatinine, the diabetic screening
assay) over ``LOINC/9318-7``, which appears with no value bins and so probably carries
little volume.

Of the medication concepts only metformin resolves: ``RxNorm/6809`` is a single
ingredient code, as is ``RxNorm/83367`` for atorvastatin.  No insulin, GLP-1 or SGLT2
ingredient code is in the vocabulary, so those exist only as branded or dose-form
products and genuinely need a set, as does the statin drug class.
"""

TASKS = {
    "At an outpatient visit, among patients with type 2 diabetes, does an HbA1c at or above 8.9 appear within 1 year, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
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
    "At an outpatient visit, among patients with type 2 diabetes, does a glucose below 54 appear within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
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
                "query": "LAB//GLUCOSE//(-inf,54)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, is diabetic ketoacidosis recorded within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
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
                "query": "DIAGNOSIS//DIABETIC_KETOACIDOSIS",
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
    "At an outpatient visit, among patients with type 2 diabetes, is stage 3 chronic kidney disease recorded within 3 years?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "3y",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1095,
                "forced_answer": False,
            },
            {
                "query": "DIAGNOSIS//CKD_STAGE_3",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1095,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, is diabetic retinopathy recorded within 3 years?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "3y",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1095,
                "forced_answer": False,
            },
            {
                "query": "DIAGNOSIS//DIABETIC_RETINOPATHY",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1095,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, is a diabetic foot ulcer recorded within 2 years?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "2y",
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
                "query": "DIAGNOSIS//DIABETIC_FOOT_ULCER",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 730,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, is a lower extremity amputation recorded within 3 years?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "3y",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1095,
                "forced_answer": False,
            },
            {
                "query": "PROCEDURE//LOWER_EXTREMITY_AMPUTATION",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1095,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, is heart failure first recorded within 3 years?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "3y",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1095,
                "forced_answer": False,
            },
            {
                "query": "DIAGNOSIS//HEART_FAILURE",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1095,
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
    "At an outpatient visit, among patients with type 2 diabetes, does insulin start within 2 years?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "2y",
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
                "query": "MEDICATION//START//INSULIN",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 730,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, does a GLP-1 receptor agonist start within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
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
                "query": "MEDICATION//START//GLP1_AGONIST",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, does an SGLT2 inhibitor start within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
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
                "query": "MEDICATION//START//SGLT2_INHIBITOR",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
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
    "At an outpatient visit, among patients with type 2 diabetes, is a retinal examination performed within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
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
                "query": "PROCEDURE//RETINAL_EXAM",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, is a statin prescribed within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
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
                "query": "MEDICATION//STATIN",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, given that an SGLT2 inhibitor is prescribed within 30 days, does an HbA1c at or above 8.9 appear within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
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
                "query": "MEDICATION//SGLT2_INHIBITOR",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": True,
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
    "At an outpatient visit, among patients with type 2 diabetes, given that an SGLT2 inhibitor is prescribed within 30 days, is a genitourinary infection recorded within 180 days, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "180d",
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
                "query": "MEDICATION//SGLT2_INHIBITOR",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": True,
            },
            {
                "query": "DIAGNOSIS//GENITOURINARY_INFECTION",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 180,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, given that an SGLT2 inhibitor is prescribed within 30 days, is diabetic ketoacidosis recorded within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
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
                "query": "MEDICATION//SGLT2_INHIBITOR",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": True,
            },
            {
                "query": "DIAGNOSIS//DIABETIC_KETOACIDOSIS",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with type 2 diabetes, given that a GLP-1 receptor agonist is prescribed within 30 days, does an HbA1c at or above 8.9 appear within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "1y",
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
                "query": "MEDICATION//GLP1_AGONIST",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": True,
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
    "At an outpatient visit, among patients with type 2 diabetes, given that insulin is prescribed within 30 days, does a glucose below 54 appear within 180 days, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "type 2 diabetes",
            "horizon": "180d",
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
                "query": "MEDICATION//INSULIN",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": True,
            },
            {
                "query": "LAB//GLUCOSE//(-inf,54)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 180,
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
