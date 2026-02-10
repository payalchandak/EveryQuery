from meds_torchdata.extensions.lightning_datamodule import Datamodule as MEDSDatamodule


class Datamodule(MEDSDatamodule):
    def predict_dataloader(self):
        return self.test_dataloader()
