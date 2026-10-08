"""Datamodule whose train loader survives a mid-epoch resume."""

from functools import cached_property

from meds import held_out_split, train_split, tuning_split
from meds_torchdata.extensions.lightning_datamodule import Datamodule
from torchdata.stateful_dataloader import StatefulDataLoader


class ResumableDatamodule(Datamodule):
    """``Datamodule`` with a stateful train loader for exact mid-epoch resume.

    ``StatefulDataLoader`` implements ``state_dict``/``load_state_dict``, which Lightning's
    fit loop (>=2.5) detects and snapshots into every checkpoint.  On ``do_resume`` the
    sampler position and RNG are restored, so a single-epoch run continues on exactly the
    examples it hadn't seen yet instead of a fresh shuffle over all of them.

    Also adds ``dataset_kwargs``: upstream constructs the dataset as ``data_class(config,
    split=...)`` with no room for per-dataset options, which would force a subclass per
    feature.  Feature toggles that live on the *dataset* (currently ``strip_delta_tokens``
    for RoPE time) are passed through here instead, so one dataset class serves every
    combination of features.
    """

    def __init__(self, *args, dataset_kwargs: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_kwargs = dict(dataset_kwargs or {})

    @cached_property
    def train_dataset(self):
        return self.data_class(self.config, split=train_split, **self.dataset_kwargs)

    @cached_property
    def val_dataset(self):
        return self.data_class(self.config, split=tuning_split, **self.dataset_kwargs)

    @cached_property
    def test_dataset(self):
        return self.data_class(self.config, split=held_out_split, **self.dataset_kwargs)

    def train_dataloader(self):
        return StatefulDataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate,
            shuffle=True,
            **self.shared_dataloader_kwargs,
        )
