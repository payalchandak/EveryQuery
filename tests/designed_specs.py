"""Helper for writing designed query-sequence specs (``sequences_path`` files) in tests.

A designed entry must spell out all six keys; this fills in the defaults a test does not care about, so a
fixture states only what it is about.
"""


def entry(
    query: str,
    duration_days: float | None = None,
    *,
    bound_event: str | None = None,
    start_event: str | None = None,
    start_duration_days: float | None = 0,
    forced_answer: bool | None = None,
) -> dict:
    """One fully spelled-out designed entry.

    Examples:
        >>> entry("A", 30)
        {'query': 'A', 'start_event': None, 'start_duration_days': 0, 'bound_event': None,
         'duration_days': 30, 'forced_answer': None}
        >>> entry("A", bound_event="DISCHARGE", start_event="ADMIT", forced_answer=True)
        {'query': 'A', 'start_event': 'ADMIT', 'start_duration_days': None, 'bound_event': 'DISCHARGE',
         'duration_days': None, 'forced_answer': True}
    """
    return {
        "query": query,
        "start_event": start_event,
        "start_duration_days": None if start_event is not None else start_duration_days,
        "bound_event": bound_event,
        "duration_days": duration_days,
        "forced_answer": forced_answer,
    }


def from_triples(sequences: dict[str, list[list]]) -> dict[str, list[dict]]:
    """Spell out compact ``[code, days]`` / ``[code, -1, bound_event]`` test literals as full entries.

    Examples:
        >>> from_triples({"s": [["A", 10], ["B", -1, "END"]]})["s"][1]["bound_event"]
        'END'
    """
    return {
        name: [entry(q[0], bound_event=q[2]) if len(q) == 3 else entry(q[0], q[1]) for q in seq]
        for name, seq in sequences.items()
    }
