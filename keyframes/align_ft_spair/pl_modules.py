from typing import Any, Optional, Literal
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim

import pytorch_lightning as pl

import models.pl_module as pl_module
import models.lr_schedulers as lr_schedulers

from timm.data.loader import MultiEpochsDataLoader

from keyframes.align_ft_spair import dataset, model

from typing import List
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class PLData(pl.LightningDataModule):
    """
    Tutorial on Lightning Data Modules:
    https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html
    """

    def __init__(
        self,
        spair_data_folder: str,
        embds_folder_dino: str,
        embds_folder_sd: str,
        img_files_np_path_train: str,
        img_files_np_path_eval: str,
        kpt_indices: List[int],
        category="aeroplane",
        img_size=960,
        embd_size=60,
        pad=True,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        # dataset params
        self.spair_data_folder = spair_data_folder
        self.embds_folder_dino = embds_folder_dino
        self.embds_folder_sd = embds_folder_sd
        self.img_files_np_path_train = img_files_np_path_train
        self.img_files_np_path_eval = img_files_np_path_eval
        self.kpt_indices = kpt_indices
        self.category = category
        self.img_size = img_size
        self.embd_size = embd_size
        self.pad = pad

        # dataloader params
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # no need to download anything
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        # TODO only load data if needed
        self.stage = stage

        # dataset
        self.train_set = dataset.SPairEmbdDataset(
            spair_data_folder=self.spair_data_folder,
            embds_folder_dino=self.embds_folder_dino,
            embds_folder_sd=self.embds_folder_sd,
            img_files_np_path=self.img_files_np_path_train,
            kpt_indices=self.kpt_indices,
            category=self.category,
            mode="train",
            img_size=self.img_size,
            embd_size=self.embd_size,
            pad=self.pad,
        )

        # TODO create val, test, and predict dataset
        self.val_set = dataset.SPairEmbdDataset(
            spair_data_folder=self.spair_data_folder,
            embds_folder_dino=self.embds_folder_dino,
            embds_folder_sd=self.embds_folder_sd,
            img_files_np_path=self.img_files_np_path_eval,
            kpt_indices=self.kpt_indices,
            category=self.category,
            mode="eval",
            img_size=self.img_size,
            embd_size=self.embd_size,
            pad=self.pad,
        )

        self.predict_dataset = dataset.SPairEmbdDataset(
            spair_data_folder=self.spair_data_folder,
            embds_folder_dino=self.embds_folder_dino,
            embds_folder_sd=self.embds_folder_sd,
            img_files_np_path=self.img_files_np_path_eval,
            kpt_indices=self.kpt_indices,
            category=self.category,
            mode="eval",
            img_size=self.img_size,
            embd_size=self.embd_size,
            pad=self.pad,
        )

    def train_dataloader(self):
        return MultiEpochsDataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # TODO maybe only load data that is needed
        return super().on_before_batch_transfer(batch, dataloader_idx)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        pass


class PLModel(pl_module.PlModule):
    model: model.FtWeightModel

    def __init__(
        self,
        # model params
        dim=1280,
        # optimzer params
        optimizer_lr: float = 1.3e-4,
        optimizer_warmup: int = 250,
        optimizer_max_iters: int = 1200000,
        optimizer_start_iter: int = 0,
    ):
        # NOTE make it compatible with LightGlue model
        model_args = {
            "dim": dim
        }
        optimizer_args = {
            "lr": optimizer_lr,
            "warmup": optimizer_warmup,
            "max_iters": optimizer_max_iters,
            "start_iter": optimizer_start_iter,
        }
        super().__init__(model_args, optimizer_args)
        self.save_hyperparameters()  # saves all arguments to model checkpoints => makes it easy to load model later

    def init_model(self, model_args) -> nn.Module:
        return model.FtWeightModel(**model_args)

    def init_optimizers(self, optimizer_args):
        optimizer = optim.AdamW(self.parameters(), lr=optimizer_args["lr"])

        # lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=optimizer_args["milestones"],
        #     gamma=optimizer_args["gamma"],
        # )

        # optimizer = optim.Adam(self.parameters(), lr=optimizer_args["lr"])
        # Apply lr scheduler per step
        lr_scheduler = lr_schedulers.CosineWarmupScheduler(
            optimizer,
            warmup=optimizer_args["warmup"],
            max_iters=optimizer_args["max_iters"],
            start_iter=(
                optimizer_args["start_iter"] if "start_iter" in optimizer_args else 0
            ),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        kpt_embds_1, kpt_embds_2 = batch[:2]
        kpt_embds_1 = self.model(kpt_embds_1)
        kpt_embds_2 = self.model(kpt_embds_2)
        return kpt_embds_1, kpt_embds_2, batch

    def get_loss(self, batch, mode: str):
        kpt_embds_1, kpt_embds_2 = batch[:2]
        kpt_embds_1 = self.model(kpt_embds_1)
        kpt_embds_2 = self.model(kpt_embds_2)
        loss = self.model.loss(kpt_embds_1, kpt_embds_2)
        # ft1_hat, ft2_hat = self.model(ft1, ft2)
        # loss = self.model.loss(ft1_hat, ft2_hat, is_pos_pair)

        self.log(f"{mode}_loss", loss)
        return loss

    # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
    #     ft1, ft2, ft3 = batch[:3]
    #     ft1 = self.model(ft1)
    #     ft2 = self.model(ft2)
    #     ft3 = self.model(ft3)
    #     return ft1, ft2, ft3, batch

    # def get_loss(self, batch, mode: str):
    #     ft1, ft2, ft3 = batch[:3]
    #     ft1 = self.model(ft1)
    #     ft2 = self.model(ft2)
    #     ft3 = self.model(ft3)
    #     loss = self.model.loss(ft1, ft2, ft3)
    #     # ft1_hat, ft2_hat = self.model(ft1, ft2)
    #     # loss = self.model.loss(ft1_hat, ft2_hat, is_pos_pair)

    #     self.log(f"{mode}_loss", loss)
    #     return loss
