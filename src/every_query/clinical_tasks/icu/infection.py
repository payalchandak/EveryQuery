"""ICU antibiotic-exposure tasks, anchored at ICU hour 24.

Nine agents, each asked at 24 hours.  Guard pair matches ``organ_support``:
``MEDS_DEATH`` implements "conditional on surviving the window" and ``TIMELINE//END``
catches a record running out while the patient is alive.

Targets are the literal ``MEDICATION//START//<name>`` order-start codes.  Note this is
the order start, which is a larger population than the ``//Administered`` event, since
an order can be placed and never given.

Where the build carries an agent under more than one name, the task takes the dominant
one only.  This drops roughly 8% of exposed subjects for piperacillin-tazobactam
(``Piperacillin-Tazobactam Na``), 9% for levofloxacin (the ``LevoFLOXacin`` casing), and
20% for ciprofloxacin (plain ``Ciprofloxacin`` alongside ``Ciprofloxacin HCl``).
Vancomycin needs no exclusion rule: ``Vancomycin Oral Liquid`` is a separate code and an
exact match on ``Vancomycin`` already leaves it out.

Imipenem and ertapenem are absent from this build under every alias, so they carry no
task.
"""

TASKS = {
    "At ICU hour 24, does piperacillin-tazobactam administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "mimic",
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "piperacillin-tazobactam",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDICATION//START//Piperacillin-Tazobactam",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does cefepime administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "mimic",
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "cefepime",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDICATION//START//CefePIME",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does ceftazidime administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "mimic",
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "ceftazidime",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDICATION//START//CefTAZidime",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does meropenem administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "mimic",
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "meropenem",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDICATION//START//Meropenem",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does intravenous vancomycin administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "mimic",
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "intravenous vancomycin",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDICATION//START//Vancomycin",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does linezolid administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "mimic",
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "linezolid",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDICATION//START//Linezolid",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does daptomycin administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "mimic",
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "daptomycin",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDICATION//START//Daptomycin",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does levofloxacin administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "mimic",
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "levofloxacin",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDICATION//START//Levofloxacin",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does ciprofloxacin administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "dataset": "mimic",
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "ciprofloxacin",
        },
        "query": [
            {
                "query": "TIMELINE//END",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDS_DEATH",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": False,
            },
            {
                "query": "MEDICATION//START//Ciprofloxacin HCl",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
}
