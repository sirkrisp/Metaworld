"""Normal estimation"""
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

from keyframes.align_ft_spair import dataset
from keyframes.align_ft_spair.utils import geo_utils, spair_utils, ft_align_utils, depth_match_utils, geom_utils, img_utils, kpt_likelihood_utils, kpt_likelihood_opt_v2, focal_opt_utils, peak_extraction_utils
from keyframes.align_ft_spair import pl_modules
from keyframes.align_ft_spair.ext import projection_network, utils_correspondence, utils_dataset
from utils import torch_utils, plotly_utils

# params
spair_data_folder="/media/user/ssd2t/datasets2/SPair-71k"
embds_folder_geo="/media/user/EXTREMESSD/datasets/SPair-71k/geo"
embds_folder_sd="/media/user/EXTREMESSD/datasets/SPair-71k/DiffusionFeatures60x60"
img_files_np_path_train="/home/user/Documents/projects/diffusion-features/experiments/ft_align/aeroplane_train_files_unique.npy"
img_files_np_path_eval="/home/user/Documents/projects/diffusion-features/experiments/ft_align/aeroplane_eval_files_unique.npy"
# depth_folder = "/media/user/EXTREMESSD/datasets/SPair-71k/miragold"
depth_folder = "/media/user/EXTREMESSD/datasets/SPair-71k/DepthAnythingV2"
masks_folder = "/media/user/EXTREMESSD/datasets/SPair-71k/SegAnyMasks"

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
kpt_features_avg_train = []
kpt_features_attn_avg_train = []
kpt_features_attn_sd_train = []
for kpt_idx in range(30):
    kpt_features = kpt_idx_to_kpt_embds_train[kpt_idx]
    # kpt_features is only None if keypoint label does not exist for object category
    if kpt_features is None:
        break

    if kpt_idx > 3 and kpt_idx < 22:
        kpt_idx_2 = kpt_idx + 1 if kpt_idx % 2 == 0 else kpt_idx - 1
        kpt_features_2 = kpt_idx_to_kpt_embds_train[kpt_idx_2]
        assert kpt_features_2 is not None, f"kpt_features is None for kpt_idx={kpt_idx_2}"
        kpt_features = torch.cat([kpt_features, kpt_features_2], dim=0)

    kpt_features_avg = torch.mean(kpt_features, dim=0, keepdim=True)  # (1, C)
    # compute dot product between kpt_features and kpt_features_avg
    kpt_features_attn = torch.bmm(kpt_features.unsqueeze(0), kpt_features_avg.unsqueeze(2)).squeeze(0)
    kpt_features_attn_avg = torch.mean(kpt_features_attn, dim=0, keepdim=True)
    kpt_features_attn_sd = torch.std(kpt_features_attn, dim=0, keepdim=True)
    kpt_features_avg_train.append(kpt_features_avg)
    kpt_features_attn_avg_train.append(kpt_features_attn_avg)
    kpt_features_attn_sd_train.append(kpt_features_attn_sd)

kpt_features_avg_train = torch.cat(kpt_features_avg_train, dim=0)
kpt_features_attn_avg_train = torch.cat(kpt_features_attn_avg_train, dim=0)
kpt_features_attn_sd_train = torch.cat(kpt_features_attn_sd_train, dim=0)
# print(kpt_features_avg_train.shape)

# 1.1) load depth
depths_train = []
for img_file in img_files_train:
    fn = os.path.basename(img_file).split(".")[0]
    # depth_file = f"{depth_folder}/{category}/depth_npy/{fn}_pred.npy"
    depth_file = f"{depth_folder}/{category}/{fn}_depth.npy"
    depth = np.load(depth_file)
    # NEW for depth any: normalize and adjust z
    depth = 1 - 0.3*(depth/np.max(depth))

    # NOTE instead of downsizing depth we will resize and upscale attentions
    depths_train.append(depth)

depths_eval = []
for img_file in img_files_eval:
    fn = os.path.basename(img_file).split(".")[0]
    # depth_file = f"{depth_folder}/{category}/depth_npy/{fn}_pred.npy"
    depth_file = f"{depth_folder}/{category}/{fn}_depth.npy"
    depth = np.load(depth_file)
    # NEW for depth any: normalize and adjust z
    depth = 1 - 0.3*(depth/np.max(depth))
    depths_eval.append(depth)

# 1.3) reproject to point clouds
xyz_train = []
for depth in depths_train:
    v0 = geom_utils.reproject_depth(depth, focal_length=5)
    xyz_train.append(v0)

xyz_eval = []
for depth in depths_eval:
    v0 = geom_utils.reproject_depth(depth, focal_length=5)
    xyz_eval.append(v0)

# load segmentation masks
seg_masks_train = []
seg_auto_masks_train = []
for img_file in img_files_train:
    fn = os.path.basename(img_file).split(".")[0]
    seg_mask_file = f"{masks_folder}/{category}/{fn}_masks.npy"
    seg_mask = np.load(seg_mask_file)
    seg_masks_train.append(seg_mask)

    seg_auto_mask_file = f"{masks_folder}/{category}/{fn}_auto_masks_seg.npy"
    seg_auto_mask = np.load(seg_auto_mask_file)
    seg_auto_masks_train.append(seg_auto_mask)

seg_masks_eval = []
seg_auto_masks_eval = []
for img_file in img_files_eval:
    fn = os.path.basename(img_file).split(".")[0]
    seg_mask_file = f"{masks_folder}/{category}/{fn}_masks.npy"
    seg_mask = np.load(seg_mask_file)
    seg_masks_eval.append(seg_mask)

    seg_auto_mask_file = f"{masks_folder}/{category}/{fn}_auto_masks_seg.npy"
    seg_auto_mask = np.load(seg_auto_mask_file)
    seg_auto_masks_eval.append(seg_auto_mask)


# ===================
# Load train ratios mean and angles mean
# ===================

cos_angles_mean = np.load("/home/user/Documents/projects/Metaworld/keyframes/align_ft_spair/notebooks/focal_length_opt_cos_angles_mean.npy")
ratios_mean = np.load("/home/user/Documents/projects/Metaworld/keyframes/align_ft_spair/notebooks/focal_length_opt_ratios_mean.npy")
cos_angles_mean = torch.from_numpy(cos_angles_mean)
ratios_mean = torch.from_numpy(ratios_mean)


# ===================
# Optimization
# ===================

# input args
query_xy_all_normalized, depth_values_all, query_xyz_list, masks, kpt_coords_corrected_all = peak_extraction_utils.correct_all_keypoint_coords(
    img_seg_masks=seg_masks_eval,
    kpt_img_coords=kpt_img_coords_eval,
    img_xyz_orig=xyz_eval
)

img_shapes = torch.zeros((query_xy_all_normalized.shape[0], 2))
for i in range(query_xy_all_normalized.shape[0]):
    img_shapes[i,:] = torch.tensor(xyz_eval[i].shape[:2])

query_xy_all_normalized_no_nan = query_xy_all_normalized.clone()
query_xy_all_normalized_no_nan[torch.isnan(query_xy_all_normalized_no_nan)] = 0.0
depth_values_all_no_nan = depth_values_all.clone()
depth_values_all_no_nan[torch.isnan(depth_values_all_no_nan)] = 0.0

vertex_mask = focal_opt_utils.generate_is_not_nan_mask(query_xy_all_normalized)


focal_lengths_inv_opt_eval = torch.ones(img_shapes.shape[0])/5

kpt_xy_normalized=query_xy_all_normalized_no_nan
kpt_depth=depth_values_all_no_nan
kpt_is_not_nan=vertex_mask
img_shapes=img_shapes
n_iter=1000
lr=0.001

# compute angle and ratio mask
angles_mask, ratios_mask = focal_opt_utils.generate_angle_and_ratio_mask(kpt_is_not_nan)

# 3) define energy function with current mean values
def energy_func(focal_lenghts_inv):
    return focal_opt_utils.focal_length_energy_mean_dist(
        focal_lenghts_inv=focal_lenghts_inv,
        img_shapes=img_shapes,
        kpt_xy_normalized=kpt_xy_normalized,
        kpt_depth=kpt_depth,
        angles_mask=angles_mask,
        ratios_mask=ratios_mask,
        cos_angles_mean=cos_angles_mean,
        ratios_mean=ratios_mean,
    )[0]

# 4) run local optimization loop
focal_lengths_inv_opt_eval, total_energy = focal_opt_utils.update_focal_lengths(
    focal_lenghts_inv=focal_lengths_inv_opt_eval,
    energy_func=energy_func,
    lr=lr,
    n_iter=n_iter
)

out_dir = "/home/user/Documents/projects/Metaworld/keyframes/align_ft_spair/notebooks"
np.save(f"{out_dir}/focal_lengths_inv_opt_v2_eval.npy", torch_utils.to_np_array(focal_lengths_inv_opt_eval))


# very similar results to the simple optimisation procedure
# focal_lengths_inv_opt_v3 = focal_opt_utils.optimize_focal_length_global_local(
#     kpt_xy_normalized=query_xy_all_normalized_no_nan,
#     kpt_depth=depth_values_all_no_nan,
#     kpt_is_not_nan=vertex_mask,
#     img_shapes=img_shapes,
#     n_global_iter=10,
#     n_local_iter=100,
#     lr=0.001
# )


# compute statistics and visualize

# reproject all points with optimised focal length
# kpts_xyz_all = focal_opt_utils.reproject_kpts(
#     focal_lenghts_inv=torch.abs(focal_lengths_inv_opt_v2),
#     img_shapes=img_shapes,
#     kpt_xy_normalized=query_xy_all_normalized_no_nan,
#     kpt_depth=depth_values_all_no_nan,
# )

# # compute statistics
# angles_mask, ratios_mask = focal_opt_utils.generate_angle_and_ratio_mask(vertex_mask)
# cos_angles, angles, ratios = kpt_likelihood_utils.compute_angles_and_ratios_parallel(kpts_xyz_all)
# cos_angles_mean, cos_angles_var, ratios_mean, ratios_var = kpt_likelihood_utils.compute_stats_over_images(
#     cos_angles, ratios, angles_mask, ratios_mask
# )

# # save results
# results = {
#     "focal_lengths_inv_opt_v2": focal_lengths_inv_opt_v2,
#     "cos_angles_mean": cos_angles_mean,
#     "cos_angles_var": cos_angles_var,
#     "ratios_mean": ratios_mean,
#     "ratios_var": ratios_var
# }
# out_dir = "/home/user/Documents/projects/Metaworld/keyframes/align_ft_spair/notebooks"
# for k, v in results.items():
#     np.save(f"{out_dir}/focal_length_opt_{k}.npy", torch_utils.to_np_array(v))