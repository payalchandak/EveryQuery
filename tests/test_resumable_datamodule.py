"""The train loader must be resumable: Lightning-stateful, and a state_dict round-trip
must continue on exactly the not-yet-seen examples (no repeats, no gaps)."""

from lightning.fabric.utilities.types import _Stateful
from torch.utils.data import Dataset

from every_query.data.datamodule import ResumableDatamodule


class _Indices(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx

    @staticmethod
    def collate(batch):
        return batch


def _make_datamodule():
    D = ResumableDatamodule.__new__(ResumableDatamodule)
    D.batch_size = 2
    D.num_workers = None
    D.pin_memory = None
    D.persistent_workers = None
    D.prefetch_factor = None
    D.__dict__["train_dataset"] = _Indices()  # bypass the cached_property
    return D


def test_train_loader_is_lightning_stateful():
    # This is the exact protocol check Lightning's fit loop uses to decide whether
    # to snapshot/restore the loader in checkpoints.
    assert isinstance(_make_datamodule().train_dataloader(), _Stateful)


def test_resume_continues_on_remaining_examples_only():
    loader = _make_datamodule().train_dataloader()
    it = iter(loader)
    seen = [*next(it), *next(it)]  # 2 of 5 batches, then "crash"
    state = loader.state_dict()

    resumed = _make_datamodule().train_dataloader()
    resumed.load_state_dict(state)
    remaining = [x for batch in resumed for x in batch]

    assert sorted(seen + remaining) == list(range(10))
