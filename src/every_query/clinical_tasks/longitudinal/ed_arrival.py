"""Longitudinal tasks anchored at ED arrival plus 2 hours.

**Columbia NYP, not MIMIC.**  See ``outpatient_visit.py`` for why the two datasets share
no codes.

Code substitutions against the Columbia vocabulary (21,264 codes):

- ``HOSPITAL_ADMISSION`` -> ``Visit/IP``, the OMOP inpatient visit concept.
- ``PROCEDURE//INTUBATION`` -> ``CPT4/31500``, emergency endotracheal intubation.
- ``MEDICATION//RBC_TRANSFUSION`` -> ``ICD10PCS/30233N1``, transfusion of nonautologous
    red blood cells into a peripheral vein.

.. warning::

    ``Visit/IP`` **undercounts admissions from the ED**, which is exactly the population
    these tasks anchor on: Columbia routes emergency-to-inpatient through ``Visit/ERIP``,
    a separate concept.  Capturing both needs a union.  This bites harder here than in
    ``hospital_discharge.py``, since every subject anchored at ED arrival who is admitted
    is a candidate for the ``ERIP`` path.

    Intubation also exists as ``ICD10PCS/0BH17EZ``, insertion of an endotracheal airway.
    Taking ``CPT4/31500`` forfeits it.  The CPT code is the emergency-procedure billing
    code and so is the better fit for an ED anchor, but which the data actually carries
    cannot be checked here.

    Red cells transfused by any route other than a peripheral vein, or autologous units,
    carry different ICD10PCS codes and are not captured.

Three specified tasks are absent.  ICU admission has no Columbia equivalent.  An
operating room procedure would need a derived set over the 359 ICD10PCS and 1,483 CPT4
codes.  ED discharge to home asks which disposition attaches to a single event, the same
shape that could not be expressed for MIMIC hospital dispositions, and Columbia carries
no ED discharge code at all: an ED discharge is the end of a ``Visit/ER``.
"""

TASKS = {
    "At ED arrival plus 2 hours, does hospital admission occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "ED arrival plus 2 hours",
            "horizon": "12h",
            "outcome": "hospital admission",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": False,
            },
            {
                "query": "Visit/IP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ED arrival plus 2 hours, does intubation occur within 6 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "ED arrival plus 2 hours",
            "horizon": "6h",
            "outcome": "intubation",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.25,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.25,
                "forced_answer": False,
            },
            {
                "query": "CPT4/31500",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.25,
                "forced_answer": None,
            },
        ],
    },
    "At ED arrival plus 2 hours, does red blood cell transfusion occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "ED arrival plus 2 hours",
            "horizon": "12h",
            "outcome": "red blood cell transfusion",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": False,
            },
            {
                "query": "ICD10PCS/30233N1",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
}
