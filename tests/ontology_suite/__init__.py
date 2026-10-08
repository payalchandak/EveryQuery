"""Independent golden fixtures, oracle and correctness suite for the ontology/eval path.

Nothing in this package may import a production labeling or closure helper.  The whole point
is that the expected answers are derived from a *separately written* reading of the semantics,
so a test cannot pass by sharing a bug with the code under test.

The one production import that is allowed is the thing being *tested* (imported inside the test
modules, never inside :mod:`oracle` or :mod:`golden`).
"""
