"""Stage 1': ``QuerySequenceDistribution`` — the sequence draw layered over the Stage 1 query draw.

The sequence analogue of ``test_stage1_query_distribution.py``.  ``QuerySequenceDistribution``
subclasses ``QueryDistribution``: the inherited ``sample`` supplies the flat ``(code, duration)``
stream, and this class adds *structure* — sequence lengths, the two sweep knobs
(``eos_first_fraction`` / ``duration_mode``), and event bounds.

Covered here, and nowhere else in the suite:

* the four ``__post_init__`` config guards (``min_queries``, ``max_queries``,
  ``eos_first_fraction``, ``duration_mode``) and the cross-field guard that ``eos_first_fraction``
  cannot force a code that is absent from the sampling universe.  Only the fifth,
  ``eventbound_fraction``, is covered elsewhere (``tests/test_event_bounded.py``);
* the independence of the structure generator from the query generator;
* the configured *length range* actually being spanned.  Every other caller of this class in the
  suite fixes ``min_queries == max_queries``, so the length draw itself is exercised only here;
* the ``duration_mode="nondecreasing"`` branch.  The class doctest only demonstrates ``"same"``.

These tests were lost when the conditional query-sequence pipeline was deleted; the class they
cover was not — it is Stage 1' of the Hydra ``run`` that forms the sampler differential's
training-pipeline arm.
"""

import numpy as np
import pytest

from every_query.data.query_seq_dataset import EOS_CODE
from every_query.generate_tasks.query_sequence_labeling import QuerySequenceDistribution
from every_query.generate_tasks.sample_tasks import QueryDistribution


def _seq_dist(**overrides) -> QuerySequenceDistribution:
    """The reference config, with ``overrides`` applied.

    Kept deliberately valid so that in :func:`test_sequence_distribution_rejects_bad_config` the
    *only* thing that can raise is the override under test.
    """
    kwargs = {
        "query_codes": ["A", "B", "C", EOS_CODE],
        "min_duration": 1.0,
        "max_duration": 365.0,
        "duration_distribution": "log-uniform",
        "min_queries": 1,
        "max_queries": 5,
    }
    kwargs.update(overrides)
    return QuerySequenceDistribution(**kwargs)


def test_the_reference_config_is_itself_valid():
    """Anchors the rejection cases below: an unmodified ``_seq_dist()`` must construct cleanly.

    Without this, every ``pytest.raises`` below could be passing because the *base* kwargs became
    invalid — the vacuous-parametrize failure mode.
    """
    dist = _seq_dist()
    assert (dist.min_queries, dist.max_queries) == (1, 5)
    assert dist.duration_mode == "random"


def test_structure_rng_does_not_perturb_the_query_draw():
    """Changing only the structure seed must leave the code/duration stream untouched.

    The two axes are independent by construction; if they ever share a generator, a change to a
    sweep knob would silently move the query distribution too — and with it the parity with
    ``sample_tasks`` that lets a sequence model and a single-query model be compared at all.

    The seed-invariance clause alone is *not* enough, and the reason is worth stating: it needs a
    fixed length to keep the two totals comparable, but ``rng.integers(n, n + 1)`` consumes no
    generator state, so moving the length draw onto ``query_rng`` would be invisible to it — the
    structure seed would simply become a no-op and both streams would still agree, just both
    wrong.  The second clause is what bites: at a *variable* length the length draw really does
    consume state, so pinning the flattened stream to ``QueryDistribution.sample`` for the same
    generator fails the moment the two axes share one.  That equality is also the parity anchor
    the whole two-sampler comparison rests on.
    """
    dist = _seq_dist(min_queries=3, max_queries=3)  # fixed length => same total either way
    a = dist.sample_sequences(20, np.random.default_rng(0), np.random.default_rng(1))
    b = dist.sample_sequences(20, np.random.default_rng(0), np.random.default_rng(99))
    assert [q for s in a for q in s] == [q for s in b for q in s]

    varied = _seq_dist(min_queries=1, max_queries=5)
    flat = [
        q for s in varied.sample_sequences(50, np.random.default_rng(0), np.random.default_rng(1)) for q in s
    ]
    assert (
        len({len(s) for s in varied.sample_sequences(50, np.random.default_rng(0), np.random.default_rng(1))})
        > 1
    )
    base = QueryDistribution(varied.query_codes, 1.0, 365.0, "log-uniform")
    assert flat == base.sample(len(flat), np.random.default_rng(0))


def test_sequence_lengths_span_the_configured_range():
    """``L ~ Uniform{min_queries..max_queries}`` — inclusive at both ends.

    An off-by-one in the length draw silently narrows the sweep: the run still succeeds and the
    labels are still valid, they just never contain a max-length sequence.
    """
    seqs = _seq_dist(min_queries=2, max_queries=4).sample_sequences(
        200, np.random.default_rng(0), np.random.default_rng(1)
    )
    assert {len(s) for s in seqs} == {2, 3, 4}


def test_duration_mode_nondecreasing_sorts_within_each_sequence():
    """``duration_mode="nondecreasing"`` must sort the horizons inside every sequence.

    Four queries per sequence so an unsorted draw is overwhelmingly unlikely to come out ordered by chance;
    the codes keep their positions and only the durations are permuted.
    """
    seqs = _seq_dist(min_queries=4, max_queries=4, duration_mode="nondecreasing").sample_sequences(
        20, np.random.default_rng(0), np.random.default_rng(1)
    )
    assert seqs and all(len(s) == 4 for s in seqs)
    for s in seqs:
        durations = [q.duration_days for q in s]
        assert durations == sorted(durations)
    # Not vacuous: the same draw without the knob is not already sorted.
    plain = _seq_dist(min_queries=4, max_queries=4).sample_sequences(
        20, np.random.default_rng(0), np.random.default_rng(1)
    )
    assert any([q.duration_days for q in s] != sorted(q.duration_days for q in s) for s in plain)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_queries": 0}, "min_queries must be >= 1"),
        ({"min_queries": 3, "max_queries": 2}, "must be >= min_queries"),
        ({"eos_first_fraction": 1.5}, r"must be in \[0, 1\]"),
        ({"duration_mode": "sorted"}, "duration_mode must be one of"),
        ({"query_codes": ["A"], "eos_first_fraction": 0.5}, "not in query_codes"),
    ],
)
def test_sequence_distribution_rejects_bad_config(kwargs, match):
    """Config errors must fail at construction, not three stages downstream.

    Each ``match`` names the specific guard, so a case cannot pass by tripping a *different*
    guard — the way a rejection parametrize quietly goes vacuous.
    """
    with pytest.raises(ValueError, match=match):
        _seq_dist(**kwargs)
