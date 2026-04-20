"""Prediction-output schema.

``PredictionSchema`` extends ``TaskQuerySchema`` (the inference *input* schema defined in
``every_query.data.schema``) with the model's two-head probability outputs:
``censor_prob`` and ``occurs_prob``.  Both probabilities are always emitted regardless of
whether the ground-truth ``boolean_value`` is available — inference is schema-driven, not
label-driven.

See #129 for the post-refactor discussion on generalizing ``occurs_prob`` → ``label_prob``
once non-occurrence tasks land.  Staying with ``occurs_prob`` for now so the schema column
matches the model's ``occurs_head`` / ``batch.occurs`` wiring end-to-end.
"""

import pyarrow as pa
from flexible_schema import Required

from every_query.data.schema import TaskQuerySchema


class PredictionSchema(TaskQuerySchema):
    """EveryQuery prediction-output row.

    Inherits every column from :class:`every_query.data.schema.TaskQuerySchema` —
    ``subject_id``, ``prediction_time``, ``query``, ``duration_days``, plus the inherited
    optional label columns (``boolean_value`` nullable, per ``LabelSchema``).  Adds:

    Attributes:
        censor_prob: Model-reported probability that the row is censored (observation
            ended before we could see whether the event fired).  ``sigmoid`` of the
            censor head's logit.
        occurs_prob: Model-reported probability that the event occurred in the duration
            window, *conditional on the row not being censored*.  ``sigmoid`` of the
            occurs head's logit.  The occurs head is loss-masked on censored rows during
            training, so ``occurs_prob`` is unconstrained when ``censor_prob`` is high —
            downstream metrics should either filter on predicted-not-censored or use
            ``censor_prob`` as a mask.

    Examples:
        A prediction-only row (no ground truth) validates:

        >>> from datetime import datetime
        >>> import pyarrow as pa
        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 30.0,
        ...      "censor_prob": 0.12, "occurs_prob": 0.73},
        ... ])
        >>> aligned = PredictionSchema.align(data)
        >>> [f.name for f in aligned.schema]
        ['subject_id', 'prediction_time', 'query', 'duration_days', 'censor_prob', 'occurs_prob']

        A row with the inherited ``boolean_value`` label filled in (eval-time shape) also
        validates:

        >>> data = pa.Table.from_pylist([
        ...     {"subject_id": 1, "prediction_time": datetime(2023, 1, 1),
        ...      "query": "ICD//I10", "duration_days": 30.0, "boolean_value": True,
        ...      "censor_prob": 0.05, "occurs_prob": 0.91},
        ... ])
        >>> set(f.name for f in PredictionSchema.align(data).schema) >= {
        ...     "subject_id", "prediction_time", "query", "duration_days",
        ...     "boolean_value", "censor_prob", "occurs_prob",
        ... }
        True
    """

    censor_prob: Required(pa.float32(), nullable=False)
    occurs_prob: Required(pa.float32(), nullable=False)
