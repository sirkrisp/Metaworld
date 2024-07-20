from enum import unique
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import torch
from keyframes.align_ft_spair.utils import geom_utils, ft_align_utils, img_utils, spair_utils
from utils import torch_utils
from tqdm import tqdm


def detect_and_thicken_edges(depth_map, canny_threshold1=50, canny_threshold2=150, dilation_kernel_size=3):
    # Apply Canny edge detection
    edges = cv2.Canny(depth_map, canny_threshold1, canny_threshold2)
    
    # Create a kernel for dilation
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    
    # Dilate the edges
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    
    return thick_edges


def erode_mask(mask: np.ndarray, kernel_size=2, iterations=2):
    # Define the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion
    eroded_mask = np.array(cv2.erode(mask.astype(np.uint8)*255, kernel, iterations=iterations)).astype(bool)
    return eroded_mask


def correct_keypoint_coords(img_seg_mask: np.ndarray, kpt_coords_np: np.ndarray):
    """For each point in kpt_coords, check if it is in the segmentation mask. If not, find the closest point in the mask and replace the point with it."""
    # edges = peak_extraction_utils.detect_and_thicken_edges((depth*255).astype(np.uint8), canny_threshold1=5, canny_threshold2=10, dilation_kernel_size=2)
    # mask = np.logical_and(img_seg_mask > 0, edges == 0)
    mask = erode_mask(img_seg_mask)
    mask_indices = np.argwhere(mask)
    # permute x and y
    mask_indices = mask_indices[:, [1, 0]]

    # Calculate distances using broadcasting
    distances = np.sqrt(((mask_indices[:,None,:] - kpt_coords_np[None,:,:]) ** 2).sum(axis=2))  # shape (num_mask_points, num_keypoints)

    # Find the index of the closest point in the mask for each keypoint
    closest_indices = np.argmin(distances, axis=0)
    kpt_coords_corrected = mask_indices[closest_indices]

    return kpt_coords_corrected, mask


def correct_all_keypoint_coords(
    img_seg_masks: List[np.ndarray],
    kpt_img_coords: List[torch.Tensor],
    img_xyz_orig: List[np.ndarray],
):
    """
    Out:
        query_xy_all_normalized: (n_imgs, n_max_kpts, 2) NOTE xy is normalized to [0,1] (division by max(img_width, img_height))
        depth_values_all: (n_imgs, n_max_kpts)
    """
    n_imgs = len(img_xyz_orig)
    n_max_kpts = np.max([torch.max(kpt_img_coords[i][:,2]).item() for i in range(n_imgs)]) + 1
    kpt_coords_corrected_all = []
    kpt_coords_corrected_normalized_all = torch.zeros((n_imgs, n_max_kpts, 2), dtype=torch.float32)
    kpt_coords_corrected_normalized_all[:, :, :] = torch.nan
    depth_values_all = torch.zeros((n_imgs, n_max_kpts), dtype=torch.float32)
    depth_values_all[:, :] = torch.nan
    kpt_xyz_list = []
    masks = []
    for i in tqdm(range(n_imgs)):
        kpt_coords = kpt_img_coords[i][:,:2]
        kpt_ids = kpt_img_coords[i][:,2]
        kpt_coords_corrected, mask = correct_keypoint_coords(img_seg_masks[i][2], kpt_coords.numpy())
        kpt_coords_corrected_all.append(kpt_coords_corrected)
        masks.append(mask)
        kpt_coords_corrected_normalized = kpt_coords_corrected / max(img_xyz_orig[i].shape[:2])
        kpt_coords_corrected_normalized_all[i,kpt_ids] = torch_utils.to_torch_tensor(kpt_coords_corrected_normalized, device="cpu").float()

        kpt_xyz = img_xyz_orig[i][kpt_coords_corrected[:,1], kpt_coords_corrected[:,0]]
        kpt_xyz_list.append(kpt_xyz)

        depth_values = kpt_xyz[:, 2]
        depth_values_all[i,kpt_ids] = torch_utils.to_torch_tensor(depth_values, device="cpu").float()

    return kpt_coords_corrected_normalized_all, depth_values_all, kpt_xyz_list, masks, kpt_coords_corrected_all


def extract_candidates_for_kpt(
    img_xyz_orig: np.ndarray,
    img_seg_mask: np.ndarray,
    img_likelihood_of_kpt: torch.Tensor,
    max_num_peaks=3,
    min_distance=10,
    min_peak_value=0.01,
    peak_rel_threshold=0.2
):
    # 1) extract peaks (candidates)
    h_orig, w_orig = img_xyz_orig.shape[:2]
    embd_size = img_likelihood_of_kpt.shape[-1]
    img_likelihood_of_kpt_up = img_utils.inv_pad_resize_img(img_likelihood_of_kpt, h_orig, w_orig)
    peak_xy_coords, peak_values = ft_align_utils.extract_local_maxima(img_likelihood_of_kpt_up, k=max_num_peaks, min_distance=min_distance)
    if peak_xy_coords is None or peak_values is None:
        return torch.empty((0,3)), torch.empty((0,2)), torch.empty((0)), torch.empty((0,2))
    # TODO also check if peak_values are too low
    peak_mask = peak_values > (torch.max(peak_values) * peak_rel_threshold)
    peak_mask = peak_mask & (peak_values > min_peak_value)
    peak_values = peak_values[peak_mask]
    if len(peak_values) == 0:
        return torch.empty((0,3)), torch.empty((0,2)), torch.empty((0)), torch.empty((0,2))
    peak_xy_coords = peak_xy_coords[peak_mask.numpy()]

    # 2) correct peak xy values
    peak_xy_coords_corrected, _ = correct_keypoint_coords(img_seg_mask, peak_xy_coords)

    # 3) only keep peaks that are different from each other
    peak_xy_coords_corrected, row_indices = np.unique(peak_xy_coords_corrected, axis=0, return_index=True)
    peak_values = peak_values[torch_utils.to_torch_tensor(row_indices, device="cpu")]
    
    # 4) extract xyz values
    peak_xyz = img_xyz_orig[peak_xy_coords_corrected[:,1], peak_xy_coords_corrected[:,0]]

    # 5) transform peak img coords to embd coords
    peak_embd_coords_xy = spair_utils.transform_image_coords_parallel(
        img_coords=peak_xy_coords_corrected,
        img_orig_height=h_orig,
        img_orig_width=w_orig,
        img_new_size=embd_size
    )
    
    # 6) convert to tensors
    peak_xyz = torch_utils.to_torch_tensor(peak_xyz, device="cpu").float()
    peak_xy_coords_corrected = torch_utils.to_torch_tensor(peak_xy_coords_corrected, device="cpu").int()
    peak_values = torch_utils.to_torch_tensor(peak_values, device="cpu").float()
    peak_embd_coords_xy = torch_utils.to_torch_tensor(peak_embd_coords_xy, device="cpu").int()

    return peak_xyz, peak_xy_coords_corrected, peak_values, peak_embd_coords_xy


def extract_candidates(
    img_xyz_orig: np.ndarray,
    img_seg_mask: np.ndarray,
    img_kpt_label_likelihood: torch.Tensor,
    min_distance=10,
    min_peak_value=0.01,
    max_candidates=30,
    peak_rel_threshold=0.2,
):
    """
    Args:
        - img_xyz_orig: (H_orig, W_orig, 3)
        - img_seg_mask: (H_orig, W_orig)
        - img_kpt_label_likelihood: (K, H, W)
    """

    k = img_kpt_label_likelihood.shape[0]

    candidates_xyz_list = []
    candidates_xy_list = []
    candidates_kpt_likelihoods_list = []
    candidates_embd_coords_xy_list = []

    for i in range(k):
        peak_xyz, peak_xy_coords, peak_values, peak_embd_coords_xy = extract_candidates_for_kpt(
            img_xyz_orig,
            img_seg_mask,
            img_kpt_label_likelihood[i],
            max_num_peaks=max_candidates,
            min_distance=min_distance,
            min_peak_value=min_peak_value,
            peak_rel_threshold=peak_rel_threshold
        )
        candidates_xyz_list.append(peak_xyz)
        candidates_xy_list.append(peak_xy_coords)
        candidates_kpt_likelihoods_list.append(peak_values)
        candidates_embd_coords_xy_list.append(peak_embd_coords_xy)

    candidates_xyz = torch.cat(candidates_xyz_list, dim=0)
    candidates_xy = torch.cat(candidates_xy_list, dim=0)
    candidate_kpt_likelihoods = torch.cat(candidates_kpt_likelihoods_list, dim=0)
    candidates_embd_coords_xy = torch.cat(candidates_embd_coords_xy_list, dim=0)

    # only keep unique candidates
    candidates_xy, row_indices = torch_utils.unique_rows(candidates_xy.long())
    candidates_xyz = candidates_xyz[row_indices]
    candidate_kpt_likelihoods = candidate_kpt_likelihoods[row_indices]
    candidates_embd_coords_xy = candidates_embd_coords_xy[row_indices].long()

    return candidates_xyz, candidates_xy, candidate_kpt_likelihoods, candidates_embd_coords_xy

