"""ICU readmission tasks, anchored at ICU discharge."""

TASKS = {
    "At ICU discharge, does ICU readmission occur before hospital discharge, conditional on being discharged from the hospital alive?": {
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
                "forced_answer": False,
            },
            {
                "query": "ICU_ADMISSION",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": "HOSPITAL_DISCHARGE",
                "duration_days": None,
                "forced_answer": None,
            },
        ],
    },
    "At ICU discharge, does ICU readmission occur within 72 hours, conditional on being discharged from the hospital alive?": {
        "metadata": {"setting": "icu", "anchor": "ICU discharge", "horizon": "72h"},
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
                "bound_event": "HOSPITAL_DISCHARGE",
                "duration_days": None,
                "forced_answer": False,
            },
            {
                "query": "ICU_ADMISSION",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 3,
                "forced_answer": None,
            },
        ],
    },
}
