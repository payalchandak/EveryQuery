"""Longitudinal tasks anchored at hospital discharge.

**Columbia NYP, not MIMIC.**  See ``outpatient_visit.py`` for why the two datasets share
no codes.

Code substitutions against the Columbia vocabulary (21,264 codes):

- ``HOSPITAL_ADMISSION`` -> ``Visit/IP``, the OMOP inpatient visit concept.
- ``ED_ARRIVAL`` -> ``Visit/ER``.

.. warning::

    ``Visit/IP`` **undercounts readmissions.**  Columbia also carries ``Visit/ERIP``, an
    emergency-room-to-inpatient visit, which is how a large share of unplanned
    readmissions present.  Capturing both needs a union, so the three readmission tasks
    take ``Visit/IP`` alone and miss readmissions admitted through the ED.  The size of
    that loss cannot be measured here, as the Columbia event data is not on this machine.

The death tasks carry no ``TIMELINE//END`` guard, following ``icu/death.py``: death
terminates the record, so that guard would exclude exactly the patients who died.  The
specification included it; it is deliberately omitted.

Five specified tasks are absent.  ICU admission has no Columbia equivalent, as OMOP has
no ICU visit concept.  Skilled nursing facility admission and hospice enrollment were
dropped as unclear: their only candidates, ``CMS Place of Service/31`` and ``/34``, are
place-of-service attributes recording where a service was delivered, not admission or
enrollment events.  Primary care and specialty visits cannot be distinguished: the
vocabulary holds only nine NUCC facility taxonomies (case management, generic
clinic/center, endoscopy, infusion therapy, MRI, radiology, mammography, oncology and
radiation oncology), none of which is primary care, and no provider-specialty attribute
is attached to visits.
"""

TASKS = {
    "At hospital discharge, does hospital readmission occur within 7 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital discharge",
            "horizon": "7d",
            "outcome": "hospital readmission",
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
    "At hospital discharge, does hospital readmission occur within 30 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital discharge",
            "horizon": "30d",
            "outcome": "hospital readmission",
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
                "query": "Visit/IP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 30,
                "forced_answer": None,
            },
        ],
    },
    "At hospital discharge, does hospital readmission occur within 90 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital discharge",
            "horizon": "90d",
            "outcome": "hospital readmission",
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
    "At hospital discharge, does an ED visit occur within 30 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital discharge",
            "horizon": "30d",
            "outcome": "ED visit",
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
    "At hospital discharge, does death occur within 30 days?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital discharge",
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
    "At hospital discharge, does death occur within 1 year?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital discharge",
            "horizon": "1y",
            "outcome": "death",
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
}
