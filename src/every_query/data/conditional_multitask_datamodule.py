"""Lightning datamodule for the all-vocabulary multitask model: training labels in, QuerySeq grids out.

:class:`~every_query.model.conditional_multitask_ar_model.ConditionalMultitaskARModel` is trained on
one artifact and evaluated on another, and the two are **independent roots** on disk:

``config.task_labels_dir`` - the TRAINING multitask labels
    The ``out_dir`` of ``EQ_generate_multitask_sequences``: per split, ``MultitaskBoundarySchema``
    metadata parquets, row-aligned packed ``.labels.npy`` sidecars and a ``_multitask_manifest.json``.
    Read by :class:`~every_query.data.multitask_dataset.MultitaskBoundaryPytorchDataset`, which unpacks
    ``(B, K, V)`` dense targets per batch for the all-vocabulary loss.

``eval_tasks_dir`` - the QuerySeq EVALUATION grid
    The ``eval/`` root written by ``EQ_generate_evaluation_query_sequences``:
    ``QuerySeqSchema`` parquets whose rows are ordered scalar
    queries with their answers.  Read by
    :class:`~every_query.data.multitask_eval_dataset.QuerySeqMultitaskEvalDataset`, which turns each
    row into ``n - 1`` teacher-forced conditioning pairs plus one scored final query.  No manifest, no
    packed labels, no sidecar.

Which loop reads which::

    Lightning loop     split                  dataset                           batch
    ----------------   --------------------   -------------------------------   ----------------------
    fit (train)        train                  MultitaskBoundaryPytorchDataset   MultitaskBoundaryBatch
    fit (validation)   tuning                 MultitaskBoundaryPytorchDataset   MultitaskBoundaryBatch
    test               held_out               QuerySeqMultitaskEvalDataset      MultitaskEvalBatch
    predict            ``predict_split``      QuerySeqMultitaskEvalDataset      MultitaskEvalBatch

The first two rows come from ``config.task_labels_dir``, the last two from ``eval_tasks_dir``.

Validation deliberately stays training-style.  ``tuning/loss`` - the dense all-vocabulary objective
on the tuning split of the *training* labels - is what ``ModelCheckpoint`` and ``EarlyStopping``
monitor, so it must be computable on every run, including one that has no evaluation grid yet, and
it must measure the quantity being optimized rather than a different task.  Grid-based metrics are
the job of ``trainer.test`` / ``trainer.predict`` / ``EQ_predict_multitask`` after training.

``eval_tasks_dir`` is therefore **optional**: ``fit`` never touches the test / predict datasets, and
nothing here reads the grid until one of them is asked for.  Asking without a grid root raises a
``ValueError`` naming ``eval_tasks_dir``.  The evaluation datasets are built on ``eval_config``: the
training ``config`` with only ``task_labels_dir`` swapped for ``eval_tasks_dir``, so
``tensorized_cohort_dir``, ``max_seq_len``, ``seq_sampling_strategy``, ``static_inclusion_mode`` and
``batch_mode`` are always the checkpoint's own.
"""

import dataclasses
from functools import cached_property
from pathlib import Path

from meds import held_out_split, tuning_split
from meds_torchdata.config import MEDSTorchDataConfig
from torch.utils.data import DataLoader, SequentialSampler

from every_query.data.datamodule import ResumableDatamodule
from every_query.data.multitask_dataset import MultitaskBoundaryPytorchDataset
from every_query.data.multitask_eval_dataset import QuerySeqMultitaskEvalDataset

# Splits the evaluation grid may be scored on.  ``train`` is excluded: the grid is an evaluation
# artifact and scoring must stay sequential (row ``i`` of the output is grid row ``i``), so a split
# whose loader would shuffle has no meaning here.
EVAL_SPLITS: tuple[str, ...] = (tuning_split, held_out_split)

MISSING_EVAL_TASKS_DIR_MSG = (
    "eval_tasks_dir is None: the test / predict datasets score a QuerySeq evaluation grid, which is a "
    "separate artifact from the training labels under config.task_labels_dir. Pass "
    "eval_tasks_dir=<the eval/ root written by EQ_generate_evaluation_query_sequences> "
    "(datamodule.eval_tasks_dir=/path on the command line)."
)


class ConditionalMultitaskDataModule(ResumableDatamodule):
    """``ResumableDatamodule`` over the multitask training labels, plus QuerySeq test / predict sets.

    The train / validation side is exactly :class:`~every_query.data.datamodule.ResumableDatamodule`
    with ``data_class`` pinned to
    :class:`~every_query.data.multitask_dataset.MultitaskBoundaryPytorchDataset`: a
    ``StatefulDataLoader`` train loader (mid-epoch resume) and ``dataset_kwargs`` forwarded to both
    training datasets.  The test / predict side reads the QuerySeq grid under ``eval_tasks_dir``
    through :class:`~every_query.data.multitask_eval_dataset.QuerySeqMultitaskEvalDataset` on
    :attr:`eval_config`, with a plain sequential ``DataLoader``.

    Args:
        config: The checkpoint's cohort / sequence settings.  ``config.task_labels_dir`` is the
            TRAINING multitask sampler root (``EQ_generate_multitask_sequences`` ``out_dir``).
        batch_size: Batch size for every loader.
        num_workers: As in the upstream ``Datamodule``.
        pin_memory: As in the upstream ``Datamodule``.
        persistent_workers: As in the upstream ``Datamodule``.
        prefetch_factor: As in the upstream ``Datamodule``.
        dataset_kwargs: Forwarded to the two training datasets, as in
            :class:`~every_query.data.datamodule.ResumableDatamodule`.  Of its keys only
            ``strip_delta_tokens`` and ``expected_vocab_size`` reach the evaluation adapter (the
            rest - manifest checks - have no meaning for a grid).  ``expected_vocab_size`` is the
            **cohort's** width ``V``: the training sidecars are leaf-only whether or not an ontology
            is in use, because the model widens targets to ``V_ext`` itself.  Its ``ontology_dir``
            key is the checkpoint's ``model.ontology_dir`` (interpolated from it in the shipped
            configs); it is lifted into :attr:`ontology_dir` for the evaluation adapter - where it
            makes ancestor query / start / bound names resolvable and raises the adapter's width
            check from ``V`` to the ontology's ``V_ext`` - **and** forwarded to the training
            datasets, which need it when the sampler drew ancestor-valued start / bound /
            conditioning codes (``ontology_mode`` other than ``"none"`` in their manifest).  Against
            a leaf-only labels directory the training datasets simply do not use it.
        eval_tasks_dir: Optional QuerySeq evaluation-grid ``eval/`` root
            (``EQ_generate_evaluation_query_sequences``).  Only ``trainer.test`` / ``trainer.predict``
            / ``EQ_predict_multitask`` need it; ``None`` is fine for ``fit``.
        max_windows: The checkpoint's ``model.max_windows``, forwarded to the evaluation adapter so a
            grid row longer than the model's window budget is rejected when the dataset is built
            rather than deep inside a forward pass.
        predict_split: The split the predict dataset reads; one of :data:`EVAL_SPLITS`.  The test
            dataset always reads ``held_out``.

    Attributes:
        train_dataset / val_dataset: ``MultitaskBoundaryPytorchDataset`` over ``config`` (inherited).
        test_dataset / predict_dataset: ``QuerySeqMultitaskEvalDataset`` over :attr:`eval_config`.
        eval_config: ``config`` with only ``task_labels_dir`` replaced by ``eval_tasks_dir``.
        ontology_dir: The ontology the evaluation adapter resolves names through; ``None`` without one.
        dataset_kwargs: The training datasets' keyword arguments (``ontology_dir`` included).

    Examples:
        Everything about the grid is lazy, so a datamodule without one is fully usable for ``fit``
        and only the grid-backed members refuse.  (An empty directory stands in for the cohort and
        labels here; no dataset is built.)

        >>> import tempfile
        >>> tmp = tempfile.TemporaryDirectory()
        >>> cfg = MEDSTorchDataConfig(
        ...     tensorized_cohort_dir=tmp.name, task_labels_dir=tmp.name, max_seq_len=8,
        ...     seq_sampling_strategy="to_end",
        ... )
        >>> D = ConditionalMultitaskDataModule(cfg, batch_size=4, max_windows=5)
        >>> D.eval_tasks_dir is None, D.max_windows, D.predict_split
        (True, 5, 'held_out')
        >>> D.test_dataloader()
        Traceback (most recent call last):
            ...
        ValueError: eval_tasks_dir is None: the test / predict datasets score a QuerySeq evaluation grid...
        >>> D.predict_dataset
        Traceback (most recent call last):
            ...
        ValueError: eval_tasks_dir is None: ...

        With a grid root, ``eval_config`` differs from ``config`` in ``task_labels_dir`` alone:

        >>> grid = tempfile.TemporaryDirectory()
        >>> D = ConditionalMultitaskDataModule(cfg, eval_tasks_dir=grid.name, max_windows=5)
        >>> D.eval_config.task_labels_dir == Path(grid.name)
        True
        >>> D.config.task_labels_dir == Path(tmp.name)
        True
        >>> a, b = dataclasses.asdict(D.config), dataclasses.asdict(D.eval_config)
        >>> a.pop("task_labels_dir") != b.pop("task_labels_dir") and a == b
        True

        The grid settings ride along in the saved hyperparameters next to the parent's:

        >>> {"batch_size", "config", "eval_tasks_dir", "max_windows", "predict_split"} <= set(D.hparams)
        True
        >>> D.hparams["eval_tasks_dir"] == grid.name, D.hparams["max_windows"]
        (True, 5)

        ``train`` is not a grid split:

        >>> ConditionalMultitaskDataModule(cfg, predict_split="train")
        Traceback (most recent call last):
            ...
        ValueError: predict_split must be one of ['held_out', 'tuning'], got 'train'; ...

        ``ontology_dir`` is remembered separately for the evaluation adapter and recorded in the
        hyperparameters, and still reaches the training datasets, which need it to read a labels
        directory whose start / bound / conditioning codes name ontology nodes.  ``expected_vocab_size``
        stays the cohort's leaf ``V`` either way - the target bits never widen:

        >>> D = ConditionalMultitaskDataModule(
        ...     cfg, dataset_kwargs={"expected_vocab_size": 7, "ontology_dir": "/onto"}, max_windows=5)
        >>> D.ontology_dir
        PosixPath('/onto')
        >>> D.dataset_kwargs == {"expected_vocab_size": 7, "ontology_dir": "/onto"}
        True
        >>> D.hparams["ontology_dir"]
        '/onto'
        >>> tmp.cleanup(); grid.cleanup()
    """

    def __init__(
        self,
        config: MEDSTorchDataConfig,
        batch_size: int = 32,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = None,
        dataset_kwargs: dict | None = None,
        eval_tasks_dir: str | Path | None = None,
        max_windows: int | None = None,
        predict_split: str = held_out_split,
    ):
        if predict_split not in EVAL_SPLITS:
            raise ValueError(
                f"predict_split must be one of {sorted(EVAL_SPLITS)}, got {predict_split!r}; the QuerySeq "
                "grid is an evaluation artifact scored in row order, so a shuffled split has no meaning."
            )
        # ``ontology_dir`` rides in ``dataset_kwargs`` so the shipped configs can interpolate it from
        # ``lightning_module.model.ontology_dir`` the way the scalar datamodule's does.  It is lifted
        # out (so :attr:`dataset_kwargs` and the hyperparameters keep recording it separately, and
        # the evaluation adapter can be given it explicitly) and then put back, because the training
        # datasets now need it too: their *targets* are still leaf-only, but a labels directory
        # sampled with ancestor-valued start / bound / conditioning codes can only be read through
        # the ontology that numbered them.  Against a leaf-only labels directory it is simply unused.
        training_kwargs = dict(dataset_kwargs or {})
        ontology_dir = training_kwargs.pop("ontology_dir", None)
        if ontology_dir is not None:
            training_kwargs["ontology_dir"] = ontology_dir
        super().__init__(
            config,
            data_class=MultitaskBoundaryPytorchDataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            dataset_kwargs=training_kwargs,
        )
        self.ontology_dir: Path | None = None if ontology_dir is None else Path(ontology_dir)
        self.eval_tasks_dir: Path | None = None if eval_tasks_dir is None else Path(eval_tasks_dir)
        self.max_windows: int | None = None if max_windows is None else int(max_windows)
        self.predict_split: str = predict_split
        # The parent already saved its own arguments; a second dict call merges into the same hparams
        # (Lightning updates an existing dict rather than replacing it).  Paths are stored as strings
        # so hparams.yaml stays ``yaml.safe_load``-able.
        self.save_hyperparameters(
            {
                "dataset_kwargs": dict(self.dataset_kwargs),
                "ontology_dir": None if self.ontology_dir is None else str(self.ontology_dir),
                "eval_tasks_dir": None if self.eval_tasks_dir is None else str(self.eval_tasks_dir),
                "max_windows": self.max_windows,
                "predict_split": self.predict_split,
            }
        )

    # --- evaluation side --------------------------------------------------------------------------

    @cached_property
    def eval_config(self) -> MEDSTorchDataConfig:
        """``config`` pointed at the grid: only ``task_labels_dir`` changes.

        Raises:
            ValueError: If ``eval_tasks_dir`` is ``None``.
        """
        if self.eval_tasks_dir is None:
            raise ValueError(MISSING_EVAL_TASKS_DIR_MSG)
        return dataclasses.replace(self.config, task_labels_dir=str(self.eval_tasks_dir))

    @cached_property
    def eval_vocab_size(self) -> int | None:
        """The width the evaluation adapter checks grid codes against: ``V_ext`` with an ontology, else ``V``.

        ``dataset_kwargs["expected_vocab_size"]`` is the cohort's leaf width (what the training
        sidecars are checked against).  A grid may name ancestor nodes, whose indices sit in
        ``[V, V_ext)``, so under an ontology the adapter's ceiling is the extended width instead.
        ``None`` when neither is known.
        """
        if self.ontology_dir is not None:
            from every_query.data.ontology import extended_vocab_size

            return extended_vocab_size(self.ontology_dir)
        expected_vocab_size = self.dataset_kwargs.get("expected_vocab_size")
        return None if expected_vocab_size is None else int(expected_vocab_size)

    def _eval_dataset(self, split: str) -> QuerySeqMultitaskEvalDataset:
        return QuerySeqMultitaskEvalDataset(
            self.eval_config,
            split=split,
            strip_delta_tokens=bool(self.dataset_kwargs.get("strip_delta_tokens", False)),
            expected_vocab_size=self.eval_vocab_size,
            max_windows=self.max_windows,
            ontology_dir=self.ontology_dir,
        )

    @cached_property
    def test_dataset(self) -> QuerySeqMultitaskEvalDataset:
        return self._eval_dataset(held_out_split)

    @cached_property
    def predict_dataset(self) -> QuerySeqMultitaskEvalDataset:
        if self.predict_split == held_out_split:
            return self.test_dataset
        return self._eval_dataset(self.predict_split)

    def _eval_dataloader(self, dataset: QuerySeqMultitaskEvalDataset) -> DataLoader:
        """A sequential loader: row ``i`` of the output must be grid row ``i``."""
        loader = DataLoader(
            dataset, shuffle=False, collate_fn=dataset.collate, **self.shared_dataloader_kwargs
        )
        if not isinstance(loader.sampler, SequentialSampler):
            raise RuntimeError(
                f"the evaluation loader must use SequentialSampler; got {type(loader.sampler).__name__}"
            )
        return loader

    def test_dataloader(self) -> DataLoader:
        return self._eval_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        return self._eval_dataloader(self.predict_dataset)
