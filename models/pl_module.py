
import yaml
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.nn as nn
from abc import abstractmethod

class PlModule(pl.LightningModule):

    def __init__(self, model_args, optimizer_args) -> None:
        super().__init__()
        self.model_args = model_args
        self.optimizer_args = optimizer_args
        self.model = self.init_model(model_args)

    @abstractmethod
    def init_model(self, model_args) -> nn.Module:
        pass

    @abstractmethod
    def init_optimizers(self, optimizer_args):
        pass

    @abstractmethod
    def get_loss(self, batch, mode="train"):
        pass

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return self.init_optimizers(self.optimizer_args)

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.get_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self.get_loss(batch, mode="test")

    