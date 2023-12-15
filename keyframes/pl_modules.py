from typing import Any
from PIL import Image
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim

import pytorch_lightning as pl

import models.pl_module as pl_module
import models.lr_schedulers as lr_schedulers
import models.lightglue as lightglue_train

import keyframes.dataset as metaworld_dataset


class PLData(pl.LightningDataModule):
    """
    Tutorial on Lightning Data Modules: 
    https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html
    """
    data_folder: str
    train_env_ids: list[int]
    val_env_ids: list[int]
    predict_data_folder: str
    num_samples_per_env: int
    max_keypoints: int
    eps: float

    def __init__(
        self, 
        data_folder, 
        train_env_ids,
        val_env_ids,
        data_folder_predict=None,
        batch_size=32,
        num_workers=4,
        num_samples_per_env=100,
        max_keypoints=100,
        eps=0.01
    ):
        super().__init__()
        self.data_folder = data_folder
        self.data_folder_predict = data_folder_predict
        self.train_env_ids = train_env_ids
        self.val_env_ids = val_env_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples_per_env = num_samples_per_env
        self.max_keypoints = max_keypoints
        self.eps = eps

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
        self.train_set = metaworld_dataset.MetaworldFeatureMatchingDataset(
            data_folder=self.data_folder,
            env_ids=self.train_env_ids,
            max_keypoints=self.max_keypoints,
            eps=self.eps
        )

        # TODO create val, test, and predict dataset
        self.val_set = metaworld_dataset.MetaworldFeatureMatchingDataset(
            data_folder=self.data_folder,
            env_ids=self.val_env_ids,
            max_keypoints=self.max_keypoints,
            eps=self.eps
        )

        self.predict_dataset = metaworld_dataset.MetaworldFeatureMatchingDataset(
            data_folder=self.data_folder,
            env_ids=self.val_env_ids,
            max_keypoints=self.max_keypoints,
            eps=self.eps,
            dev=True
        )


    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True, 
            drop_last=True, 
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False, 
            drop_last=False, 
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False, 
            drop_last=False, 
            num_workers=self.num_workers
        )
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False, 
            drop_last=False, 
            num_workers=self.num_workers
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
    model: lightglue_train.LightGlue

    def __init__(self, model_args, optimizer_args):
        super().__init__(model_args, optimizer_args)
        self.save_hyperparameters()  # saves all arguments to model checkpoints => makes it easy to load model later
        # self.clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
        # self.clip_model.visual.output_tokens = True
        self.image_size = (360,480)
        # self.image_size.requires_grad = False

    def init_model(self, model_args) -> nn.Module:
        return lightglue_train.LightGlue(**model_args)

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
            start_iter=optimizer_args["start_iter"] if "start_iter" in optimizer_args else 0,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pred = self.model(self.batch_to_input_data(batch))
        return pred, batch

    def get_loss(self, batch, mode: str):
        input_data = self.batch_to_input_data(batch)
        pred = self.model(input_data)
        losses, _ = self.model.loss(pred, input_data)
        loss = torch.mean(losses["total"])

        self.log(f"{mode}_loss", loss)
        return loss
    
    def batch_to_input_data(self, batch):
        if len(batch) == 14:
            img_0, img_1, depth_0, depth_1, seg_0, seg_1, match_data, keypoints_1, keypoints_2, descriptors_1, descriptors_2, matches_1, matches_2, assignment_mtr = batch
        else:
            keypoints_1, keypoints_2, descriptors_1, descriptors_2, matches_1, matches_2, assignment_mtr = batch

        # self.image_size = self.image_size.to(descriptors_1.device)

        input_data = {
            "keypoints0": keypoints_1,
            "descriptors0": descriptors_1,
            "keypoints1": keypoints_2,
            "descriptors1": descriptors_2,
            "view0": {
                "image_size": self.image_size
            },
            "view1":{
                "image_size": self.image_size
            },
            "gt_matches0": matches_1,
            "gt_matches1": matches_2,
            "gt_assignment": assignment_mtr,
        }

        return input_data
