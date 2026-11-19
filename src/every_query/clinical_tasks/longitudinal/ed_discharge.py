"""Longitudinal tasks anchored at ED discharge.

**Columbia NYP, not MIMIC.**  See ``outpatient_visit.py`` for why the two datasets share
no codes.

Code substitutions against the Columbia vocabulary (21,264 codes):

- ``ED_ARRIVAL`` -> ``Visit/ER``, the OMOP emergency room visit concept.
- ``HOSPITAL_ADMISSION`` -> ``Visit/IP``, the OMOP inpatient visit concept.
- ``OUTPATIENT_VISIT`` -> ``Visit/OP``, the OMOP outpatient visit concept.

.. warning::

    ``Visit/IP`` **undercounts admissions arriving through the ED**, which Columbia
    routes through the separate ``Visit/ERIP`` concept.  Capturing both needs a union.

The death task carries no ``TIMELINE//END`` guard, following ``icu/death.py``: death
terminates the record, so that guard would exclude exactly the patients who died.  The
specification included it; it is deliberately omitted.

The anchor is structural rather than coded: an ED discharge is the end of a ``Visit/ER``,
not an event with a code of its own.
"""

TASKS = {
    "At ED discharge, does an ED revisit occur within 72 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "ED discharge",
            "horizon": "72h",
            "outcome": "ED revisit",
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
                "query": "Visit/ER",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 3,
                "forced_answer": None,
            },
        ],
    },
    "At ED discharge, does an ED revisit occur within 30 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "ED discharge",
            "horizon": "30d",
            "outcome": "ED revisit",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": False,
            },
            {
                "query": "Visit/ER",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": None,
            },
        ],
    },
    "At ED discharge, does hospital admission occur within 7 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "ED discharge",
            "horizon": "7d",
            "outcome": "hospital admission",
        },
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
                "bound_event": None,
                "duration_days": 7,
                "forced_answer": False,
            },
            {
                "query": "Visit/IP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 7,
                "forced_answer": None,
            },
        ],
    },
    "At ED discharge, does death occur within 30 days?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "ED discharge",
            "horizon": "30d",
            "outcome": "death",
        },
        "query": [
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": None,
            },
        ],
    },
    "At ED discharge, does an outpatient visit occur within 14 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "ED discharge",
            "horizon": "14d",
            "outcome": "outpatient visit",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 14,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 14,
                "forced_answer": False,
            },
            {
                "query": "Visit/OP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 14,
                "forced_answer": None,
            },
        ],
    },
}
