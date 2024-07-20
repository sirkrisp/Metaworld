"""Generate segmentation masks"""
ROOT_DIR = "/home/user/Documents/projects/Metaworld"

import sys
sys.path.append(ROOT_DIR)

# %matplotlib widget  
import argparse
import gc
import os
import json
import random
from PIL import Image
from typing import Dict, List, Union

import cv2
import numpy as np

import gc
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import PILToTensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import importlib
from typing import List
from easydict import EasyDict as edict
from tqdm import tqdm

# visualization
import plotly.express as px
import plotly.graph_objects as go
from lightglue import viz2d
from einops import rearrange

# import open3d as o3d
# from open3d.web_visualizer import draw

from keyframes.align_ft_spair import dataset
from keyframes.align_ft_spair.utils import geo_utils, spair_utils, ft_align_utils, depth_match_utils, geom_utils, img_utils, kpt_likelihood_utils, kpt_likelihood_opt_v2, focal_opt_utils, peak_extraction_utils
from keyframes.align_ft_spair import pl_modules
from keyframes.align_ft_spair.ext import projection_network, utils_correspondence, utils_dataset
from utils import torch_utils, plotly_utils

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# params
spair_data_folder="/media/user/ssd2t/datasets2/SPair-71k"
embds_folder_geo="/media/user/EXTREMESSD/datasets/SPair-71k/geo"
embds_folder_sd="/media/user/EXTREMESSD/datasets/SPair-71k/DiffusionFeatures60x60"
img_files_np_path_train="/home/user/Documents/projects/diffusion-features/experiments/ft_align/aeroplane_train_files_unique.npy"
img_files_np_path_eval="/home/user/Documents/projects/diffusion-features/experiments/ft_align/aeroplane_eval_files_unique.npy"
depth_folder = "/media/user/EXTREMESSD/datasets/SPair-71k/miragold"
# kpt_indices=[3, 4, 5]
category="aeroplane"
img_size=960
embd_size=60
pad=True

img_files_train = np.load(img_files_np_path_train).tolist()
img_files_eval = np.load(img_files_np_path_eval).tolist()

flips_train = [False]
flips_eval = [False]

# load embeddings
img_embds_train, img_embds_hat_train = geo_utils.load_geo_embds(
    img_files_train,
    embds_folder_dino=embds_folder_geo,
    embds_folder_sd=embds_folder_sd,
    flips=flips_train,
)
img_embds_train = img_embds_train.detach().cpu()
img_embds_hat_train = img_embds_hat_train.detach().cpu()

# load embeddings
img_embds_eval, img_embds_hat_eval = geo_utils.load_geo_embds(
    img_files_eval,
    embds_folder_dino=embds_folder_geo,
    embds_folder_sd=embds_folder_sd,
    flips=flips_eval,
)
img_embds_eval = img_embds_eval.detach().cpu()
img_embds_hat_eval = img_embds_hat_eval.detach().cpu()

# build kpt_idx_to_kpt_embds
kpt_idx_to_kpt_embds_train, kpt_embd_coords_train, kpt_img_coords_train = spair_utils.build_kpt_idx_to_kpt_embds(
    img_files=img_files_train,
    img_embds_hat=img_embds_hat_train,
    spair_data_folder=spair_data_folder,
    img_size=img_size,
    embd_size=embd_size,
    pad=pad,
    flips=flips_train
)
kpt_idx_to_kpt_embds_eval, kpt_embd_coords_eval, kpt_img_coords_eval = spair_utils.build_kpt_idx_to_kpt_embds(
    img_files=img_files_eval,
    img_embds_hat=img_embds_hat_eval,
    spair_data_folder=spair_data_folder,
    img_size=img_size,
    embd_size=embd_size,
    pad=pad,
    flips=flips_eval
)

# average keypoint embeddings
# kpt_features_avg_train = []
# kpt_features_attn_avg_train = []
# kpt_features_attn_sd_train = []
# for kpt_idx in range(30):
#     kpt_features = kpt_idx_to_kpt_embds_train[kpt_idx]
#     # kpt_features is only None if keypoint label does not exist for object category
#     if kpt_features is None:
#         break

#     if kpt_idx > 3 and kpt_idx < 22:
#         kpt_idx_2 = kpt_idx + 1 if kpt_idx % 2 == 0 else kpt_idx - 1
#         kpt_features_2 = kpt_idx_to_kpt_embds_train[kpt_idx_2]
#         assert kpt_features_2 is not None, f"kpt_features is None for kpt_idx={kpt_idx_2}"
#         kpt_features = torch.cat([kpt_features, kpt_features_2], dim=0)

#     kpt_features_avg = torch.mean(kpt_features, dim=0, keepdim=True)  # (1, C)
#     # compute dot product between kpt_features and kpt_features_avg
#     kpt_features_attn = torch.bmm(kpt_features.unsqueeze(0), kpt_features_avg.unsqueeze(2)).squeeze(0)
#     kpt_features_attn_avg = torch.mean(kpt_features_attn, dim=0, keepdim=True)
#     kpt_features_attn_sd = torch.std(kpt_features_attn, dim=0, keepdim=True)
#     kpt_features_avg_train.append(kpt_features_avg)
#     kpt_features_attn_avg_train.append(kpt_features_attn_avg)
#     kpt_features_attn_sd_train.append(kpt_features_attn_sd)

# kpt_features_avg_train = torch.cat(kpt_features_avg_train, dim=0)
# kpt_features_attn_avg_train = torch.cat(kpt_features_attn_avg_train, dim=0)
# kpt_features_attn_sd_train = torch.cat(kpt_features_attn_sd_train, dim=0)
# print(kpt_features_avg_train.shape)



ckpt_path="/home/user/Documents/model_ckpts/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
sam.to("cuda")

mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

out_dir="/media/user/EXTREMESSD/datasets/SPair-71k/SegAnyMasks"
category="aeroplane"

def generate_seg_gt(img_files, kpt_img_coords):
    for img_idx in tqdm(range(len(img_files))):
        img_file = img_files[img_idx]
        img=np.array(Image.open(img_file))
        input_points=kpt_img_coords[img_idx].numpy()[:,:2]
        input_labels=np.ones(input_points.shape[0], dtype=int)
        masks = mask_generator.generate(img)

        masks_seg = [mask["segmentation"] for mask in masks]
        masks_areas = [mask["area"] for mask in masks]
        masks_stability_scores = [mask["stability_score"] for mask in masks]

        # stack
        masks_seg_np = np.stack(masks_seg, axis=0)
        masks_areas_np = np.array(masks_areas)
        masks_stability_scores_np = np.array(masks_stability_scores)

        # save
        folder = f"{out_dir}/{category}"
        # create folder if not exist
        os.makedirs(folder, exist_ok=True)
        file_id = os.path.basename(img_file).split('.')[0]
        mask_seg_file = f"{folder}/{file_id}_auto_masks_seg.npy"
        mask_areas_file = f"{folder}/{file_id}_auto_masks_areas.npy"
        mask_stability_scores_file = f"{folder}/{file_id}_auto_masks_stability_scores.npy"
        np.save(mask_seg_file, masks_seg_np)
        np.save(mask_areas_file, masks_areas_np)
        np.save(mask_stability_scores_file, masks_stability_scores_np)


generate_seg_gt(img_files_eval, kpt_img_coords_eval)