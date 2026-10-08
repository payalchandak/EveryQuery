"""The base load/getitem/collate contract of ``QuerySeqPytorchDataset``.

The query-sequence peer of ``test_dataset_logic.py`` (which covers the single-query
``EveryQueryPytorchDataset``).  Everything here is the *unconditional* contract — what holds for
any query-sequence labels directory, before any opt-in feature:

* ``encode_query`` is a plain vocabulary lookup and raises ``KeyError`` off-vocabulary;
* ``_seeded_getitem`` returns ``queries``/``durations``/``answers`` of equal length, with binary
  (never null) answers, and honours negative indices;
* ``collate`` emits ``(B, L)`` for each of ``q_codes``/``q_durations``/``q_answers``/``q_mask``,
  zero-pads the short sequences, and maps ``True``/``False`` to ``ANSWER_YES``/``ANSWER_NO`` with
  ``ANSWER_NO`` in the padding.

The other modules that touch this class (``test_queryseq_starts.py``, ``test_rope_time.py``) all
enter through an opt-in column — explicit window starts, or the RoPE delta-token strip — and
assert on *those* tensors.  Nothing was left asserting the shapes and answer classes they are
layered on top of.

The ``seq_dataset`` / ``seq_sample_batch`` fixtures (``conftest.py``) build a real dataset over
the session cohort with sequence lengths 2 and 3, so padding is exercised rather than assumed.
"""

import pytest

from every_query.data.query_seq_dataset import ANSWERS_COL, QuerySeqBatch
from every_query.model.answers import ANSWER_NO, ANSWER_YES


def test_seq_dataset_loads_and_encodes(seq_dataset):
    """The labels load, and query encoding is the cohort's own vocabulary lookup.

    An unknown code must raise rather than silently encode to PAD (0), which is a real code slot's neighbour
    and would train the model on a query nobody asked.
    """
    assert len(seq_dataset) > 0
    # EOS may be absent in the tiny test cohort; encode_query is a plain vocab lookup.
    a_code = next(iter(seq_dataset.code_to_index))
    assert seq_dataset.encode_query(a_code) == seq_dataset.code_to_index[a_code]
    with pytest.raises(KeyError):
        seq_dataset.encode_query("NOT_A_REAL_CODE")


def test_seq_dataset_getitem_carries_sequences(seq_dataset):
    """One item carries three equal-length parallel lists plus the patient window.

    The three lists are zipped positionally downstream, so a length disagreement here becomes a silently
    shifted answer rather than an error.
    """
    item = seq_dataset[0]
    assert len(item["queries"]) == len(item["durations"]) == len(item["answers"])
    assert item["answers"].dtype == bool, "answers are binary, never None"
    assert "dynamic" in item
    last = seq_dataset[-1]
    assert len(last["queries"]) == len(seq_dataset.queries[len(seq_dataset) - 1]), (
        "negative index slices offsets"
    )


def test_seq_collate_shapes_and_padding(seq_sample_batch):
    """``collate`` emits four aligned ``(B, L)`` tensors and zero-pads the short sequences.

    ``q_mask`` is the only thing separating a real query from padding; if padding carried a
    nonzero code or duration, a masked-out slot would still be a well-formed query the model could
    learn from.
    """
    batch = seq_sample_batch
    assert isinstance(batch, QuerySeqBatch)
    B = batch.batch_size
    L = batch.n_queries
    assert batch.q_codes.shape == (B, L)
    assert batch.q_durations.shape == (B, L)
    assert batch.q_answers.shape == (B, L)
    assert batch.q_mask.shape == (B, L)

    # Padded positions carry zeros / mask False; the fixture mixes lengths 2 and 3.
    lengths = batch.q_mask.sum(dim=1)
    assert lengths.min() == 2 and lengths.max() == 3
    pad = ~batch.q_mask
    assert pad.any(), "fixture must mix sequence lengths or padding is never exercised"
    assert (batch.q_codes[pad] == 0).all()
    assert (batch.q_durations[pad] == 0).all()


def test_seq_collate_answer_classes(seq_dataset, seq_sample_batch):
    """Binary answers: True -> ANSWER_YES, False -> ANSWER_NO; padding holds ANSWER_NO."""
    batch = seq_sample_batch
    raw = seq_dataset.schema_df[ANSWERS_COL].to_list()
    assert any(any(a) for a in raw) and any(not all(a) for a in raw), (
        "fixture must contain both answer classes or the mapping is untested in one direction"
    )
    for i, answers in enumerate(raw):
        for j, answer in enumerate(answers):
            expected = ANSWER_YES if answer else ANSWER_NO
            assert batch.q_answers[i, j].item() == expected, f"row {i} pos {j}"
        # padding beyond the real length is ANSWER_NO
        for j in range(len(answers), batch.n_queries):
            assert batch.q_answers[i, j].item() == ANSWER_NO
