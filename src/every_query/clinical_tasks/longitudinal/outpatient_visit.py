"""Longitudinal tasks anchored at an outpatient visit.

**Columbia NYP, not MIMIC.**  This dataset uses standard terminologies (OMOP ``Visit/*``,
ICD10CM, SNOMED, LOINC, RxNorm, CPT4), so none of its codes are interchangeable with the
MIMIC codes under ``icu/``.  Every task carries ``dataset`` in its metadata for that
reason.

Code substitutions against the Columbia vocabulary (21,264 codes):

- ``OUTPATIENT_VISIT`` -> ``Visit/OP``, the OMOP outpatient visit concept.  This is both
    the anchor and one target.
- ``ED_ARRIVAL`` -> ``Visit/ER``, the OMOP emergency room visit concept.
- ``MEDS_DEATH`` needs no substitution.

Tasks carrying a non-death target keep the ``TIMELINE//END`` guard as specified.  Its
presence in Columbia could **not** be confirmed: the available vocabulary file excludes
every code containing "TIME" by design, dropping 25 codes.

The death tasks carry no censoring guard, following ``icu/death.py``: death terminates
the record, so a ``TIMELINE//END`` guard would exclude exactly the patients who died.
The specification included that guard; it is deliberately omitted.

Two specified tasks are absent.  The ICU-admission task has no Columbia equivalent, since
OMOP has no ICU visit concept and the vocabulary contains no ICU or critical-care
encounter type; the nearest option, critical-care CPT4 (99291/99292), is a billing event
rather than an admission.  The five first-diagnosis tasks need ICD10CM code sets and are
deferred.
"""

TASKS = {
    "At an outpatient visit, does death occur within 1 year?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
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
    "At an outpatient visit, does death occur within 5 years?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "horizon": "5y",
            "outcome": "death",
        },
        "query": [
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1825,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, does an ED visit occur within 30 days, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
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
    "At an outpatient visit, does an ED visit occur within 1 year, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "horizon": "1y",
            "outcome": "ED visit",
        },
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
                "forced_answer": False,
            },
            {
                "query": "Visit/ER",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
    "At an outpatient visit, does another outpatient visit occur within 1 year, conditional on surviving the window?": {
        "metadata": {
            "dataset": "columbia",
            "setting": "longitudinal",
            "anchor": "an outpatient visit",
            "horizon": "1y",
            "outcome": "outpatient visit",
        },
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
                "forced_answer": False,
            },
            {
                "query": "Visit/OP",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 365,
                "forced_answer": None,
            },
        ],
    },
}
