"""Datamodule whose train loader survives a mid-epoch resume."""

from meds_torchdata.extensions.lightning_datamodule import Datamodule
from torchdata.stateful_dataloader import StatefulDataLoader


class ResumableDatamodule(Datamodule):
    """``Datamodule`` with a stateful train loader for exact mid-epoch resume.

    ``StatefulDataLoader`` implements ``state_dict``/``load_state_dict``, which Lightning's
    fit loop (>=2.5) detects and snapshots into every checkpoint.  On ``do_resume`` the
    sampler position and RNG are restored, so a single-epoch run continues on exactly the
    examples it hadn't seen yet instead of a fresh shuffle over all of them.
    """

    def train_dataloader(self):
        return StatefulDataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate,
            shuffle=True,
            **self.shared_dataloader_kwargs,
        )
