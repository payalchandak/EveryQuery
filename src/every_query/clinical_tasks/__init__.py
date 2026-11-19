"""Clinical prediction task definitions over MEDS-formatted EHR data.

Organized by setting, then one file per anchor family.  Each file exposes a ``TASKS``
dict whose keys are the clinical questions verbatim.  Each value has two parts:

- ``metadata``: ``setting``, ``anchor``, and the axis the task varies along, which is
    ``horizon`` for duration-bounded targets and ``boundary`` for event-bounded ones.
- ``query``: a list of sub-questions.  Entries with a fixed ``forced_answer`` are
    guards defining the denominator; the final entry (``forced_answer: None``) is the
    target.
"""
