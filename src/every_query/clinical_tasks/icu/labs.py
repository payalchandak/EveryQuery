"""ICU laboratory-derangement tasks, anchored at ICU hour 24.

Guard pair matches ``organ_support``: ``MEDS_DEATH`` implements "conditional on
surviving the window" and ``TIMELINE//END`` catches a record running out while the
patient is alive.

**Thresholds are the build's, not arbitrary.**  Lab values are binned into population
deciles as ``LAB//<itemid>//<unit>//value_[lo,hi)``, and those bins are real tokens in
the model vocabulary, so a threshold question is only answerable at a decile edge.  Each
task therefore takes the outermost bin for its analyte and states that bin's actual
boundary, chosen so the resulting question is a recognized clinical derangement in its
own right rather than an approximation of some other number.

Three candidate tasks were dropped because their outermost bin falls inside the normal
range, leaving no clinical event to predict: potassium at or above 4.9 (normal 3.5-5.0),
sodium at or above 143 (normal 135-145), and glucose below 83 (normal 70-100).
"""

TASKS = {
    "At ICU hour 24, does a lactate value at or above 4.1 appear within 12 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "12h", "analyte": "lactate"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": False,
            },
            {
                "query": "LAB//50813//mmol/L//value_[4.1,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a lactate value at or above 4.1 appear within 6 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "6h", "analyte": "lactate"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.25,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.25,
                "forced_answer": False,
            },
            {
                "query": "LAB//50813//mmol/L//value_[4.1,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.25,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does an arterial pH value below 7.26 appear within 12 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "12h", "analyte": "arterial pH"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": False,
            },
            {
                "query": "LAB//50820//units//value_[-inf,7.26)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a PaO2 value below 43 appear within 24 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "24h", "analyte": "PaO2"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "LAB//50821//mm Hg//value_[-inf,43.0)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a potassium value below 3.5 appear within 24 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "24h", "analyte": "potassium"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "LAB//50971//mEq/L//value_[-inf,3.5)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a sodium value below 133 appear within 48 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "48h", "analyte": "sodium"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": False,
            },
            {
                "query": "LAB//50983//mEq/L//value_[-inf,133.0)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a creatinine value at or above 2.3 appear within 48 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "48h", "analyte": "creatinine"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": False,
            },
            {
                "query": "LAB//50912//mg/dL//value_[2.3,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a creatinine value at or above 2.3 appear within 72 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "72h", "analyte": "creatinine"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 3,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 3,
                "forced_answer": False,
            },
            {
                "query": "LAB//50912//mg/dL//value_[2.3,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 3,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a hemoglobin value below 7.9 appear within 24 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "24h", "analyte": "hemoglobin"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "LAB//51222//g/dL//value_[-inf,7.9)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a platelet count value below 85 appear within 48 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "48h",
            "analyte": "platelet count",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": False,
            },
            {
                "query": "LAB//51265//K/uL//value_[-inf,85.0)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does an INR value at or above 2.6 appear within 48 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "48h", "analyte": "INR"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": False,
            },
            {
                "query": "LAB//51237//UNK//value_[2.6,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a glucose value at or above 191 appear within 24 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "24h", "analyte": "glucose"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "LAB//50931//mg/dL//value_[191.0,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a total bilirubin value at or above 2.6 appear within 72 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "72h",
            "analyte": "total bilirubin",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 3,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 3,
                "forced_answer": False,
            },
            {
                "query": "LAB//50885//mg/dL//value_[2.6,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 3,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a bicarbonate value below 20 appear within 24 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "24h", "analyte": "bicarbonate"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "LAB//50882//mEq/L//value_[-inf,20.0)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
}
