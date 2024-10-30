from meds_torch.data.components.pytorch_dataset import PytorchDataset
from mixins import SeedableMixin


class EveryQueryDataset(PytorchDataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

    @SeedableMixin.WithSeed
    def _seeded_getitem(self, idx):
        out = super()._seeded_getitem(idx)
        return out


# EveryQueryDataset( cfg, 'train')
