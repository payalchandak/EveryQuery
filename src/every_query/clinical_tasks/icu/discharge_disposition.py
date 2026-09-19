"""Hospital discharge-disposition tasks, anchored at ICU hour 24.

The question is simply which disposition this admission's hospital discharge carries.
Each target is a specific ``HOSPITAL_DISCHARGE//<disposition>`` code.

``bound_event: HOSPITAL_ADMISSION`` is a scoping device, not part of the clinical
question.  Hospital events alternate, so the next discharge always precedes the next
admission; bounding there confines the target to *this* admission's discharge.  Without
it a later admission's discharge also matches, inflating positives in this shard from
48 to 71 (HOME), 33 to 44 (SNF), and 8 to 15 (HOSPICE).

These carry no ``TIMELINE//END`` censoring guard.  Only 7 of 187 subjects lack a
discharge altogether, while the guard discarded every never-readmitted patient (107 of
187), taking 40 to 88% of the positives with it.  Censoring is left to the labeler
(#278).
"""

TASKS = {
    "At ICU hour 24, is the hospital discharge disposition HOME?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "disposition": "HOME"},
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
    "At ICU hour 24, is the hospital discharge disposition SKILLED NURSING FACILITY?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "disposition": "SKILLED NURSING FACILITY"},
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
    "At ICU hour 24, is the hospital discharge disposition HOSPICE?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "disposition": "HOSPICE"},
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
