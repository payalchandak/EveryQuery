import torch 
from meds import LabelSchema
from meds_torchdata import MEDSPytorchDataset, MEDSTorchDataConfig

class EveryQueryPytorchDataset(MEDSPytorchDataset, torch.utils.data.Dataset):

    LABEL_COL = LabelSchema.categorical_value_name
    
    def __init__(self, cfg: MEDSTorchDataConfig, split: str):
        super().__init__(cfg, split)


