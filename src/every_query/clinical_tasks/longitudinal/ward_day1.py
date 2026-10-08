"""Longitudinal tasks anchored at hospital day 1 on a ward.

**Columbia NYP, not MIMIC.**  See ``outpatient_visit.py`` for why the two datasets share
no codes.

Code substitutions against the Columbia vocabulary (21,264 codes):

- ``CONSULT//PALLIATIVE_CARE`` -> ``ICD10CM/Z51.5``, encounter for palliative care.
- ``CODE_STATUS//DNR`` -> ``ICD10CM/Z66``, do not resuscitate.

.. warning::

    **The anchor cannot exclude the ICU.**  Columbia has no ICU visit concept and no
    ward-level location attribute, so "on a ward" cannot be separated from any other
    inpatient location.  These tasks anchor in practice at hospital day 1 of an inpatient
    stay, ICU included.

    Both substitutes are **diagnosis codes, not orders or events**.  ``Z66`` states a
    recorded DNR status, which is close to the question.  ``Z51.5`` marks an encounter
    for palliative care rather than a consult specifically.

    Each also has a sibling in the vocabulary, ``Z51.50`` and ``Z66.00``, and taking one
    code forfeits the other.  Which variant the data actually uses cannot be checked
    here, since only the Columbia vocabulary lists are on this machine, not the events.
    If the unused sibling carries the volume, these tasks will be near-empty.

``HOSPITAL_DISCHARGE`` stays bare, as in the MIMIC files, and is still to be derived.  In
Columbia it is structural rather than coded: a hospital discharge is the end of a
``Visit/IP``, not an event with a code of its own.

The death task carries no ``TIMELINE//END`` guard, following ``icu/death.py``.

Three specified tasks are absent.  ICU admission has no Columbia equivalent.  Rapid
response team activation has no code in any standard terminology and no candidate in the
vocabulary.  An operating room procedure would need a derived set over the 359 ICD10PCS
and 1,483 CPT4 codes, deciding which are theatre procedures.
"""

TASKS = {
    "At hospital day 1 on a ward, does death occur before hospital discharge?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital day 1 on a ward",
            "boundary": "hospital discharge",
            "outcome": "death",
        },
        "query": [
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
    "At hospital day 1 on a ward, does hospital discharge occur within 48 hours, conditional on being discharged from the hospital alive?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital day 1 on a ward",
            "horizon": "48h",
            "outcome": "hospital discharge",
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
                "bound_event": "HOSPITAL_DISCHARGE",
                "duration_days": None,
                "forced_answer": False,
            },
            {
                "query": "HOSPITAL_DISCHARGE",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": None,
            },
        ],
    },
    "At hospital day 1 on a ward, does hospital discharge occur within 7 days, conditional on being discharged from the hospital alive?": {
        "metadata": {
            "placeholder": True,
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital day 1 on a ward",
            "horizon": "7d",
            "outcome": "hospital discharge",
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
                "bound_event": "HOSPITAL_DISCHARGE",
                "duration_days": None,
                "forced_answer": False,
            },
            {
                "query": "HOSPITAL_DISCHARGE",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 7,
                "forced_answer": None,
            },
        ],
    },
    "At hospital day 1 on a ward, does a palliative care consult occur within 7 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital day 1 on a ward",
            "horizon": "7d",
            "outcome": "palliative care consult",
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
                "query": "ICD10CM/Z51.5",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 7,
                "forced_answer": None,
            },
        ],
    },
    "At hospital day 1 on a ward, is a DNR code status recorded within 7 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "hospital day 1 on a ward",
            "horizon": "7d",
            "outcome": "DNR code status",
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
                "query": "ICD10CM/Z66",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 7,
                "forced_answer": None,
            },
        ],
    },
}
