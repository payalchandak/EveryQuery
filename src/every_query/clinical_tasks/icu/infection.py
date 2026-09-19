"""ICU antibiotic-exposure tasks, anchored at ICU hour 24.

Eleven agents, each asked at 24 hours.  Guard pair matches ``organ_support``:
``MEDS_DEATH`` implements "conditional on surviving the window" and ``TIMELINE//END``
catches a record running out while the patient is alive.

Every target is a ``DERIVED//ABX_*`` code whose membership is **not yet defined**.  Nine
of the eleven agents are present in the build under ``MEDICATION//*`` names, but folding
those names into a derived code requires two decisions per agent that the spec does not
make: which event forms count as administration (``//Administered`` versus
``MEDICATION//START//`` order starts, which differ by up to 17k subjects), and which
name variants belong (``Piperacillin-Tazobactam Na``, the ``LevoFLOXacin`` casing,
``Ciprofloxacin HCl``, and the exclusions of ``Vancomycin Oral Liquid`` and
``Ciprofloxacin 0.3% Ophth Soln``).

Imipenem and ertapenem do not appear in the build under any alias, so those two tasks
can never resolve to a positive.
"""

TASKS = {
    "At ICU hour 24, does piperacillin-tazobactam administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
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
                "query": "DERIVED//ABX_PIPERACILLIN_TAZOBACTAM",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does cefepime administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "24h", "intervention": "cefepime"},
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
                "query": "DERIVED//ABX_CEFEPIME",
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
                "query": "DERIVED//ABX_CEFTAZIDIME",
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
                "query": "DERIVED//ABX_MEROPENEM",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does imipenem administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "24h", "intervention": "imipenem"},
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
                "query": "DERIVED//ABX_IMIPENEM",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does ertapenem administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "ertapenem",
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
                "query": "DERIVED//ABX_ERTAPENEM",
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
                "query": "DERIVED//ABX_VANCOMYCIN_IV",
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
                "query": "DERIVED//ABX_LINEZOLID",
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
                "query": "DERIVED//ABX_DAPTOMYCIN",
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
                "query": "DERIVED//ABX_LEVOFLOXACIN",
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
                "query": "DERIVED//ABX_CIPROFLOXACIN",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
}
