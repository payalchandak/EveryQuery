"""ICU disposition tasks, anchored at ICU hour 24.

Each key is the clinical question verbatim.  Each value carries the question's
``metadata`` and its ``query``: a list of sub-questions where the entries with a fixed
``forced_answer`` are guards defining the denominator, and the final entry
(``forced_answer: None``) is the target.
"""

TASKS = {
    "At ICU hour 24, does ICU discharge occur within 24 hours, conditional on being discharged from the ICU alive?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "24h"},
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
                "bound_event": "ICU_DISCHARGE",
                "duration_days": None,
                "forced_answer": False,
            },
            {
                "query": "ICU_DISCHARGE",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does ICU discharge occur within 48 hours, conditional on being discharged from the ICU alive?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "48h"},
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
                "bound_event": "ICU_DISCHARGE",
                "duration_days": None,
                "forced_answer": False,
            },
            {
                "query": "ICU_DISCHARGE",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does ICU discharge occur within 72 hours, conditional on being discharged from the ICU alive?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "72h"},
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
                "bound_event": "ICU_DISCHARGE",
                "duration_days": None,
                "forced_answer": False,
            },
            {
                "query": "ICU_DISCHARGE",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 3,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does ICU discharge occur within 7 days, conditional on being discharged from the ICU alive?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "7d"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 7,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": "ICU_DISCHARGE",
                "duration_days": None,
                "forced_answer": False,
            },
            {
                "query": "ICU_DISCHARGE",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 7,
                "forced_answer": None,
            },
        ],
    },
}
