"""ICU organ-support tasks, anchored at ICU hour 24.

Vasopressors and airway/ventilation interventions at 12 and 24 hours, plus intermittent
hemodialysis at 48 hours and two transfusion products at 24 hours.  Both guards are load-bearing and distinct: ``MEDS_DEATH`` implements the
question's own "conditional on surviving the window", while ``TIMELINE//END`` catches a
record that runs out while the patient is alive.  The target is never death, so neither
guard can delete the positive class.

**Infusions are dose-binned, so these ask about dose, not mere exposure.**  The model
vocabulary splits every infusion into dose deciles,
``INFUSION_START//<itemid>//value_[lo,hi)``, with no unbinned token, exactly as it does
for labs.  "Does norepinephrine start at all" would need a union of ten bins, so each
infusion task instead takes the top bin and asks a question that stands on its own:
escalation to a high-dose vasopressor, or a large-volume transfusion.

======================  ==========================  ===========================
agent                   top bin starts at           plausible unit
======================  ==========================  ===========================
norepinephrine          0.320514                    mcg/kg/min
epinephrine             0.354919                    mcg/kg/min
phenylephrine           3.00137                     mcg/kg/min
vasopressin             3.6                         units/hr
dopamine                15.0093                     mcg/kg/min
red blood cells         700                         mL
platelets               705.882                     mL
======================  ==========================  ===========================

The units are **not recorded in the build** (``valueuom`` is null for all seven), so the
readings above are inferred from magnitude and are not confirmed.  The bin boundaries
themselves are exact.

Procedure codes take the build's three-part ``PROCEDURE//START//<itemid>`` form, not the
``PROCEDURE_START//<itemid>`` of the spec.  The itemids are unchanged.
"""

TASKS = {
    "At ICU hour 24, does high-dose norepinephrine administration occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "high-dose norepinephrine",
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
                "query": "INFUSION_START//221906//value_[0.32051384,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does high-dose norepinephrine administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "high-dose norepinephrine",
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
                "query": "INFUSION_START//221906//value_[0.32051384,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does high-dose epinephrine administration occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "high-dose epinephrine",
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
                "query": "INFUSION_START//221289//value_[0.35491893,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does high-dose epinephrine administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "high-dose epinephrine",
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
                "query": "INFUSION_START//221289//value_[0.35491893,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does high-dose phenylephrine administration occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "high-dose phenylephrine",
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
                "query": "INFUSION_START//221749//value_[3.0013728,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does high-dose phenylephrine administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "high-dose phenylephrine",
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
                "query": "INFUSION_START//221749//value_[3.0013728,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does high-dose vasopressin administration occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "high-dose vasopressin",
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
                "query": "INFUSION_START//222315//value_[3.6,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does high-dose vasopressin administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "high-dose vasopressin",
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
                "query": "INFUSION_START//222315//value_[3.6,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does high-dose dopamine administration occur within 12 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "12h",
            "intervention": "high-dose dopamine",
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
                "query": "INFUSION_START//221662//value_[15.009341,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 0.5,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does high-dose dopamine administration occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "high-dose dopamine",
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
                "query": "INFUSION_START//221662//value_[15.009341,inf)",
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
    "At ICU hour 24, does intermittent hemodialysis occur within 48 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "48h",
            "intervention": "intermittent hemodialysis",
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
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": False,
            },
            {
                "query": "PROCEDURE//START//225441",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 2,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does large-volume red blood cell transfusion occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "large-volume red blood cell transfusion",
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
                "query": "INFUSION_START//225168//value_[700.0,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
    "At ICU hour 24, does large-volume platelet transfusion occur within 24 hours, conditional on surviving the window?": {
        "metadata": {
            "setting": "icu",
            "anchor": "ICU hour 24",
            "horizon": "24h",
            "intervention": "large-volume platelet transfusion",
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
                "query": "INFUSION_START//225170//value_[705.8823,inf)",
                "start_event": None,
                "start_duration_days": 0,
                "bound_event": None,
                "duration_days": 1,
                "forced_answer": None,
            },
        ],
    },
}
