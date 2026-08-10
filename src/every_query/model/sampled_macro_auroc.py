"""Macro-averaged sampled AUROC as a ``torchmetrics.Metric``.

A task's AUROC equals ``P(score(pos) > score(neg))`` for a random pos/neg pair, so the
win/tie/loss indicator on a single offline-sampled pair is an unbiased (high-variance)
estimate of it.  Macro-averaging those indicators across tasks estimates macro AUROC at
``O(n_tasks)`` forward examples instead of ``O(split size)``.

Scores are compared, never calibrated, so any monotonic transform of the model output works.
The callback passes raw logits deliberately: ``sigmoid`` saturates to exactly 1.0 (logit >= 17
in fp32, >= 6.5 in bf16), which would collapse a correctly-ranked pair into a tie.

The pairs are fed in by ``TaskAurocTrackingCallback``, which owns the dataloader over the
offline-sampled pair set (see ``every_query.generate_tasks.sample_task_tracking_pairs``).
This class only owns the accumulate-and-score half.
"""

import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from every_query.data.dataset import EveryQueryBatch


class SampledMacroAUROC(Metric):
    """Accumulate scored pos/neg rows, then macro-average the per-task win/tie/loss indicator.

    A task is keyed by ``(query, duration_days)`` and contributes one indicator only if both a
    positive and a negative row were seen for it; tasks missing a class are dropped.

    Examples:
        >>> m = SampledMacroAUROC()
        >>> query = torch.tensor([10, 10, 20, 20, 30, 30])
        >>> duration = torch.tensor([7.0] * 6)
        >>> occurs = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> scores = torch.tensor([0.1, 0.9, 0.9, 0.1, 0.5, 0.5])
        >>> m.update(query, duration, occurs, scores)
        >>> out = m.compute()

        Task 10 wins (1.0), task 20 loses (0.0), task 30 ties (0.5):

        >>> float(out["auroc"]) == (1.0 + 0.0 + 0.5) / 3
        True
        >>> int(out["n_tasks"])
        3

        With nothing usable, ``auroc`` is NaN and callers skip logging on ``n_tasks == 0``:

        >>> out = SampledMacroAUROC().compute()
        >>> bool(out["auroc"].isnan()), int(out["n_tasks"])
        (True, 0)
    """

    def __init__(self):
        # sync_on_compute=False is load-bearing: the tracking pair set is tiny and identical on
        # every rank, so TaskAurocTrackingCallback scores it on rank 0 only.  A syncing compute()
        # would enter a collective that only rank 0 reaches, and deadlock.
        super().__init__(sync_on_compute=False)
        # add_state defaults to persistent=False, so none of this lands in state_dict and
        # existing checkpoints are unaffected.
        # dist_reduce_fx is declared for shape-correctness only — it never runs, because
        # sync_on_compute=False above means this metric is never synced across ranks.
        self.add_state("query", default=[], dist_reduce_fx="cat")
        self.add_state("duration", default=[], dist_reduce_fx="cat")
        self.add_state("label", default=[], dist_reduce_fx="cat")
        self.add_state("score", default=[], dist_reduce_fx="cat")

    def update(
        self,
        query: torch.Tensor,
        duration_days: torch.Tensor,
        occurs: torch.Tensor,
        occurs_scores: torch.Tensor,
    ) -> None:
        """Record one batch of scored rows, dropping out-of-vocab queries.

        ``occurs_scores`` is any monotonic score -- the callback passes logits.  Only the
        pos-vs-neg comparison is used, so calibration is irrelevant here.

        Out-of-vocab query codes all encode to ``PAD_INDEX``, so distinct OOV tasks would
        otherwise collide onto one key and corrupt the estimate.

        Examples:
            >>> m = SampledMacroAUROC()
            >>> m.update(
            ...     torch.tensor([0, 11]), torch.tensor([7.0, 7.0]),
            ...     torch.tensor([1, 1]), torch.tensor([0.5, 0.5]),
            ... )
            >>> m.query[0].tolist()
            [11]
        """
        keep = query != EveryQueryBatch.PAD_INDEX
        self.query.append(query[keep].cpu())
        self.duration.append(duration_days[keep].cpu())
        self.label.append(occurs[keep].cpu())
        self.score.append(occurs_scores[keep].cpu())

    def compute(self) -> dict[str, torch.Tensor]:
        """Macro-average the per-task indicator over tasks that saw both classes.

        Examples:
            Tasks missing a class are excluded — task 20 here has only a positive row:

            >>> m = SampledMacroAUROC()
            >>> m.update(
            ...     torch.tensor([10, 10, 20]), torch.tensor([7.0, 7.0, 7.0]),
            ...     torch.tensor([0, 1, 1]), torch.tensor([0.1, 0.9, 0.5]),
            ... )
            >>> out = m.compute()
            >>> float(out["auroc"]), int(out["n_tasks"])
            (1.0, 1)
        """
        scores_by_task: dict[tuple[int, float], dict[int, float]] = {}
        # len() rather than truthiness: after a _sync_dist these states are tensors, not lists,
        # and `if self.query:` on a multi-element tensor raises.
        if len(self.query):
            rows = zip(
                dim_zero_cat(self.query).tolist(),
                dim_zero_cat(self.duration).tolist(),
                dim_zero_cat(self.label).tolist(),
                dim_zero_cat(self.score).tolist(),
                strict=True,
            )
            for q, duration, label, score in rows:
                task_scores = scores_by_task.setdefault((q, duration), {})
                if int(label) in task_scores:
                    raise ValueError(
                        f"Task {(q, duration)} saw more than one row with occurs={int(label)}. "
                        "This metric keys one score per (task, class), so extra rows would be "
                        "silently dropped and the estimate would reflect only the last one. "
                        "sample_task_tracking_pairs emits exactly one positive and one negative "
                        "per task; if that changed to k pairs, group the scores into lists here "
                        "and average the indicator over the pos x neg combinations."
                    )
                task_scores[int(label)] = score

        indicators = []
        for task_scores in scores_by_task.values():
            if 0 not in task_scores or 1 not in task_scores:
                continue
            pos, neg = task_scores[1], task_scores[0]
            indicators.append(1.0 if pos > neg else 0.0 if pos < neg else 0.5)

        auroc = sum(indicators) / len(indicators) if indicators else float("nan")
        return {
            "auroc": torch.tensor(auroc),
            "n_tasks": torch.tensor(float(len(indicators))),
        }
