"""Macro-averaged sampled AUROC as a ``torchmetrics.Metric``.

A task's AUROC equals ``P(score(pos) > score(neg))`` for a random pos/neg pair, so the
win/tie/loss indicator on a single offline-sampled pair is an unbiased (high-variance)
estimate of it.  Macro-averaging those indicators across tasks estimates macro AUROC at
``O(n_tasks)`` forward examples instead of ``O(split size)``.

The pairs are fed in by ``TaskAurocTrackingCallback``, which owns the dataloader over the
offline-sampled pair set (see ``every_query.generate_tasks.sample_task_tracking_pairs``).
This class only owns the accumulate-and-score half.
"""

import torch
from torchmetrics import Metric

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
        >>> probs = torch.tensor([0.1, 0.9, 0.9, 0.1, 0.5, 0.5])
        >>> m.update(query, duration, occurs, probs)
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
        self.add_state("query", default=[], dist_reduce_fx="cat")
        self.add_state("duration", default=[], dist_reduce_fx="cat")
        self.add_state("label", default=[], dist_reduce_fx="cat")
        self.add_state("prob", default=[], dist_reduce_fx="cat")

    def update(
        self,
        query: torch.Tensor,
        duration_days: torch.Tensor,
        occurs: torch.Tensor,
        occurs_probs: torch.Tensor,
    ) -> None:
        """Record one batch of scored rows, dropping out-of-vocab queries.

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
        self.prob.append(occurs_probs[keep].cpu())

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
        probs_by_task: dict[tuple[int, float], dict[int, float]] = {}
        if self.query:
            rows = zip(
                torch.cat(self.query).tolist(),
                torch.cat(self.duration).tolist(),
                torch.cat(self.label).tolist(),
                torch.cat(self.prob).tolist(),
                strict=True,
            )
            for q, duration, label, prob in rows:
                probs_by_task.setdefault((q, duration), {})[int(label)] = prob

        indicators = []
        for task_probs in probs_by_task.values():
            if 0 not in task_probs or 1 not in task_probs:
                continue
            pos, neg = task_probs[1], task_probs[0]
            indicators.append(1.0 if pos > neg else 0.0 if pos < neg else 0.5)

        auroc = sum(indicators) / len(indicators) if indicators else float("nan")
        return {
            "auroc": torch.tensor(auroc),
            "n_tasks": torch.tensor(float(len(indicators))),
        }
