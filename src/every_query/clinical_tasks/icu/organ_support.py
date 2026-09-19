"""ICU organ-support tasks, anchored at ICU hour 24.

Five vasopressors and three airway/ventilation interventions, each asked at 12 and 24
hours.  Both guards are load-bearing and distinct: ``MEDS_DEATH`` enforces the
question's own "conditional on surviving the window", while ``TIMELINE//END`` catches a
record that runs out while the patient is still alive.  Unlike the mortality tasks, the
target here is not death, so neither guard can delete the positive class.

Procedure codes take the build's three-part ``PROCEDURE//START//<itemid>`` form, not the
``PROCEDURE_START//<itemid>`` of the spec; infusions really do use the two-part
``INFUSION_START//<itemid>``.  The itemids are unchanged.
"""

TASKS = {
    "At ICU hour 24, does norepinephrine administration occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "norepinephrine",
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
                "query": "INFUSION_START//221906",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does norepinephrine administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "norepinephrine",
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
                "query": "INFUSION_START//221906",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does epinephrine administration occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "epinephrine",
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
                "query": "INFUSION_START//221289",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does epinephrine administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "epinephrine",
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
                "query": "INFUSION_START//221289",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does phenylephrine administration occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "phenylephrine",
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
                "query": "INFUSION_START//221749",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does phenylephrine administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "phenylephrine",
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
                "query": "INFUSION_START//221749",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does vasopressin administration occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "vasopressin",
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
                "query": "INFUSION_START//222315",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does vasopressin administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "vasopressin",
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
                "query": "INFUSION_START//222315",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does dopamine administration occur within 12 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "12h", "intervention": "dopamine"},
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
                "query": "INFUSION_START//221662",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does dopamine administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {"setting": "icu", "anchor": "ICU hour 24", "horizon": "24h", "intervention": "dopamine"},
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
                "query": "INFUSION_START//221662",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does intubation occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "intubation",
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
                "query": "PROCEDURE//START//224385",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does intubation occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "intubation",
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
                "query": "PROCEDURE//START//224385",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does invasive ventilation occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "invasive ventilation",
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
                "query": "PROCEDURE//START//225792",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does invasive ventilation occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "invasive ventilation",
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
                "query": "PROCEDURE//START//225792",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does non-invasive ventilation occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "non-invasive ventilation",
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
                "query": "PROCEDURE//START//225794",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does non-invasive ventilation occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "non-invasive ventilation",
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
                "query": "PROCEDURE//START//225794",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
}
