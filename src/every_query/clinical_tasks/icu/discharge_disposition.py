"""Hospital discharge-disposition tasks, anchored at ICU hour 24.

Each target is a specific ``HOSPITAL_DISCHARGE//<disposition>`` code, bounded by the
next hospital admission.

These carry no ``TIMELINE//END`` censoring guard.  Most patients are never readmitted,
so the bound never fires and the censoring window runs to the end of the record, which
puts ``TIMELINE//END`` inside it by construction.  The guard therefore excluded every
never-readmitted patient, including those discharged to the destination in question who
simply never came back, who are positives.  It could not separate "the record ended
before anything happened" from "the outcome happened, then the record ended".  Censoring
is left to the labeler (#278).
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
