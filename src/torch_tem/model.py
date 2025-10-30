from lightning import LightningModule
from torch import Module

from torch_tem import Parameters


class TEMModel(LightningModule):

    def __init__(self, hparams: Parameters):
        super().__init__()
        self.save_hyperparameters(hparams.model_dump())


class Inference(Module):
    def forward():
        return


class Generative(Module):
    def forward(n):
        return
