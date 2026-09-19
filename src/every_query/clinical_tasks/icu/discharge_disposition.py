"""Hospital discharge-disposition tasks, anchored at ICU hour 24.

Each target is a specific ``HOSPITAL_DISCHARGE//<disposition>`` code, bounded by the
next hospital admission.

These carry no ``TIMELINE//END`` censoring guard.  A patient who dies before the bound
has their record end there, so such a guard would exclude them, when dying is instead
proof the discharge never happened and should score a genuine negative (#278).
"""

TASKS = {
    "At ICU hour 24, does a hospital discharge with disposition HOME occur before hospital readmission?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "boundary": "hospital readmission",
            "disposition": "HOME",
        },
        "query": [
            {
                "query": "HOSPITAL_DISCHARGE//HOME",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": "HOSPITAL_ADMISSION",
                "duration_days": None,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a hospital discharge with disposition SKILLED NURSING FACILITY occur before hospital readmission?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "boundary": "hospital readmission",
            "disposition": "SKILLED NURSING FACILITY",
        },
        "query": [
            {
                "query": "HOSPITAL_DISCHARGE//SKILLED NURSING FACILITY",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": "HOSPITAL_ADMISSION",
                "duration_days": None,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does a hospital discharge with disposition HOSPICE occur before hospital readmission?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "boundary": "hospital readmission",
            "disposition": "HOSPICE",
        },
        "query": [
            {
                "query": "HOSPITAL_DISCHARGE//HOSPICE",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": "HOSPITAL_ADMISSION",
                "duration_days": None,
                "forced_answer": None,
            },
        ],
    },
}
