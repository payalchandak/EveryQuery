"""Longitudinal tasks anchored at hospital discharge.

**Columbia NYP, not MIMIC.**  See ``outpatient_visit.py`` for why the two datasets share
no codes.

Code substitutions against the Columbia vocabulary (21,264 codes):

- ``HOSPITAL_ADMISSION`` -> ``Visit/IP``, the OMOP inpatient visit concept.
- ``ED_ARRIVAL`` -> ``Visit/ER``.
- ``SNF_ADMISSION`` -> ``CMS Place of Service/31``.
- ``HOSPICE_ENROLLMENT`` -> ``CMS Place of Service/34``.

.. warning::

    Two of those four shift meaning and should be reviewed.

    ``Visit/IP`` **undercounts readmissions.**  Columbia also carries ``Visit/ERIP``, an
    emergency-room-to-inpatient visit, which is how a large share of unplanned
    readmissions present.  Capturing both needs a union, so the three readmission tasks
    take ``Visit/IP`` alone and miss readmissions admitted through the ED.  The size of
    that loss cannot be measured here, as the Columbia event data is not on this machine.

    The two CMS codes are **place-of-service attributes, not admission events**.  They
    mark that a service was delivered at a skilled nursing facility or under hospice,
    which is the nearest available signal but is not the same as an admission or an
    enrollment.  No alternative exists in the vocabulary.

The death tasks carry no ``TIMELINE//END`` guard, following ``icu/death.py``: death
terminates the record, so that guard would exclude exactly the patients who died.  The
specification included it; it is deliberately omitted.

Three specified tasks are absent.  ICU admission has no Columbia equivalent, as OMOP has
no ICU visit concept.  Primary care and specialty visits cannot be distinguished: the
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
    "At hospital discharge, does skilled nursing facility admission occur within 7 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital discharge",
            "horizon": "7d",
            "outcome": "skilled nursing facility admission",
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
                "query": "CMS Place of Service/31",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 7,
                "forced_answer": None,
            },
        ],
    },
    "At hospital discharge, does hospice enrollment occur within 90 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital discharge",
            "horizon": "90d",
            "outcome": "hospice enrollment",
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
                "query": "CMS Place of Service/34",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 90,
                "forced_answer": None,
            },
        ],
    },
}
