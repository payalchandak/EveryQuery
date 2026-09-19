"""ICU mortality tasks.

Duration-bounded first (does death occur within H), then event-bounded (does death
occur before a boundary event).  Anchored at ICU hour 24 except where the question
says otherwise.
"""

TASKS = {
    "At ICU hour 24, does death occur within 24 hours?": {
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
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does death occur within 72 hours?": {
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
                "bound_event": None,
                "duration_days": 3,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does death occur within 28 days?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "28d"},
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 28,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 28,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does death occur within 90 days?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "90d"},
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
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does death occur within 1 year?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "1y"},
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
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does death occur before ICU discharge?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "boundary": "ICU discharge"},
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
                "bound_event": "ICU_DISCHARGE",
                "duration_days": None,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does death occur before hospital discharge?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "boundary": "hospital discharge"},
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
                "bound_event": "HOSPITAL_DISCHARGE",
                "duration_days": None,
                "forced_answer": None,
            },
        ],
    },
    "At ICU discharge, does death occur before hospital discharge?": {
        "metadata": {"setting": "icu", "anchor": "ICU discharge", "boundary": "hospital discharge"},
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
                "bound_event": "HOSPITAL_DISCHARGE",
                "duration_days": None,
                "forced_answer": None,
            },
        ],
    },
}
