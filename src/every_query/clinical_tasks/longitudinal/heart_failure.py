"""Longitudinal heart failure tasks, anchored at an outpatient visit among patients with
heart failure.

**Columbia NYP, not MIMIC.**  See ``outpatient_visit.py`` for why the two datasets share
no codes.

The heart failure restriction is carried by the anchor spec, not by the queries.

.. warning::

    **12 of these 21 tasks are placeholders and cannot be run.**  They keep the
    specification's concept names and carry ``"placeholder": True`` in metadata.  Eleven
    await derived code sets.  The twelfth, ICU admission, is different in kind: Columbia
    has **no ICU visit concept at all**, so unlike the drug and diagnosis sets it may not
    be derivable from this dataset.

Resolved substitutions:

- ``HOSPITAL_ADMISSION`` -> ``Visit/IP``, which undercounts admissions arriving through
    the ED, routed separately as ``Visit/ERIP``.
- ``ED_ARRIVAL`` -> ``Visit/ER``.
- ``LAB//NATRIURETIC_PEPTIDE`` -> ``LOINC/33762-6``, NT-proBNP, matching the question.

Lab thresholds take the outermost decile bin and state its real boundary:

- creatinine above 2.0 -> ``>= 2.34``, significant renal dysfunction.
- potassium above 5.5 -> ``>= 5.0``.  This is the conventional hyperkalemia flag rather
    than the moderate-hyperkalemia cutoff asked for, and it is the more apt threshold for
    a heart failure cohort, where RAAS inhibitors and MRAs make potassium monitoring
    routine at exactly that level.

The last four tasks form two treatment-comparison pairs, each contrasting a drug
prescribed against the same drug not prescribed.  The negative arm is the library's first
conditioning guard with ``forced_answer: False`` **and a window of its own**, 30 days
against a 1-year target, which is how it differs from a censoring guard.
"""

TASKS = {
    "At an outpatient visit, among patients with heart failure, does a hospitalization occur within 90 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
            "horizon": "90d",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 90,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 90,
                "forced_answer": False,
            },
            {
                "query": "Visit/IP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 90,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, does a hospitalization occur within 1 year, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "Visit/IP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, does an ED visit occur within 90 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
            "horizon": "90d",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 90,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 90,
                "forced_answer": False,
            },
            {
                "query": "Visit/ER",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 90,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, does an ICU admission occur within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "ICU_ADMISSION",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, does a creatinine at or above 2.34 appear within 1 year, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "LOINC/2160-0//UCUM/mg/dL//value_[2.34,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, does a potassium at or above 5.0 appear within 1 year, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "LOINC/2823-3//UCUM/mmol/L//value_[5.0,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, is atrial fibrillation recorded within 3 years?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "DIAGNOSIS//ATRIAL_FIBRILLATION",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1095,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, is stage 3 chronic kidney disease recorded within 3 years?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
    "At an outpatient visit, among patients with heart failure, does ICD or CRT implantation occur within 2 years?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "PROCEDURE//ICD_OR_CRT_IMPLANTATION",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 730,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, does death occur within 1 year?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
            "horizon": "1y",
        },
        "query": [
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, does death occur within 5 years?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
    "At an outpatient visit, among patients with heart failure, is an ACE inhibitor, ARB or ARNI prescribed within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "MEDICATION//RAAS_INHIBITOR",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, is a beta blocker prescribed within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "MEDICATION//BETA_BLOCKER",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, is an SGLT2 inhibitor prescribed within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, is a mineralocorticoid receptor antagonist prescribed within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "MEDICATION//MRA",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, is a potassium measured within 90 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
            "horizon": "90d",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 90,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 90,
                "forced_answer": False,
            },
            {
                "query": "LOINC/2823-3",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 90,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, is a NT-proBNP measured within 1 year, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "LOINC/33762-6",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, given that an SGLT2 inhibitor is prescribed within 30 days, does a hospitalization occur within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "Visit/IP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, given that no SGLT2 inhibitor is prescribed within 30 days, does a hospitalization occur within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "forced_answer": False,
            },
            {
                "query": "Visit/IP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, given that a beta blocker is prescribed within 30 days, does a hospitalization occur within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "MEDICATION//BETA_BLOCKER",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": True,
            },
            {
                "query": "Visit/IP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, among patients with heart failure, given that no beta blocker is prescribed within 30 days, does a hospitalization occur within 1 year, conditional on surviving the window?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "cohort": "heart failure",
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
                "query": "MEDICATION//BETA_BLOCKER",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": False,
            },
            {
                "query": "Visit/IP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
}
