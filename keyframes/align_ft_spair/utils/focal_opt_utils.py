from typing import Callable, Optional, List
from click import Option
import torch
import numpy as np
from keyframes.align_ft_spair.utils import (
    ft_align_utils,
    spair_utils,
    geom_utils,
    kpt_likelihood_utils,
)
from utils import torch_utils
from tqdm import tqdm
from sklearn.cluster import KMeans


def compute_kpt_xy_and_depth(
    kpt_img_coords: List[torch.Tensor],
    img_xyz_orig: List[np.ndarray],
    img_embds_hat: List[torch.Tensor],
    kpt_features_avg: torch.Tensor,
    kpt_features_attn_avg: torch.Tensor,
    kpt_features_attn_sd: torch.Tensor,
    thresh_rate=0.95,
    window_size=20,
):
    """
    Out:
        query_xy_all_normalized: (n_imgs, n_max_kpts, 2) NOTE xy is normalized to [0,1] (division by max(img_width, img_height))
        depth_values_all: (n_imgs, n_max_kpts)
    """
    n_imgs = len(img_xyz_orig)
    n_max_kpts = kpt_features_avg.shape[0]
    query_xy_all_normalized = torch.zeros((n_imgs, n_max_kpts, 2), dtype=torch.float32)
    depth_values_all = torch.zeros((n_imgs, n_max_kpts), dtype=torch.float32)
    query_xy_all_normalized[:, :, :] = torch.nan
    depth_values_all[:, :] = torch.nan
    query_xyz_list = []
    for i in tqdm(range(n_imgs)):
        # TODO compute img_kpt_label_likelihood
        img_xyz_orig_torch_i = torch_utils.to_torch_tensor(
            img_xyz_orig[i], device="cpu"
        )
        query_xy = kpt_img_coords[i][:, 0:2]
        query_kpt_labels = kpt_img_coords[i][:, 2].int()
        img_kpt_label_likelihood = kpt_likelihood_utils.compute_kpt_label_likelihood(
            img_embd_hat=img_embds_hat[i],
            kpt_embd_hat_mean=kpt_features_avg,
            kpt_attn_mean=kpt_features_attn_avg[:, 0],
            kpt_attn_sd=kpt_features_attn_sd[:, 0],
        )
        query_xyz = ft_align_utils.kpt_img_coords_to_xyz_with_correction(
            kpt_img_coords_xy=query_xy,
            kpt_labels=kpt_img_coords[i][:, 2],
            img_xyz_orig=img_xyz_orig_torch_i,
            img_kpt_label_likelihood=img_kpt_label_likelihood,
            thresh_rate=thresh_rate,
            window_size=window_size,
        )
        query_xyz_list.append(query_xyz)
        K = torch_utils.to_torch_tensor(
            geom_utils.get_calibration_matrix(*img_xyz_orig[i].shape[:2]), device="cpu"
        )
        query_xy_corrected = geom_utils.project_to_img_plane(query_xyz, K=K)
        # query_xy_corrected_index = query_xy_corrected * max(img_xyz_orig[i].shape[:2])
        # query_xy_corrected_index = query_xy_corrected_index.int()
        # print(i, query_xy_corrected, sep="\n")
        # depth_values = img_xyz_orig_torch_i[query_xy_corrected_index[:,1], query_xy_corrected_index[:,0],2]
        depth_values = query_xyz[:, 2]

        query_xy_all_normalized[i, query_kpt_labels, :] = query_xy_corrected.float()
        depth_values_all[i, query_kpt_labels] = depth_values.float()
    return query_xy_all_normalized, depth_values_all, query_xyz_list


def reproject_kpts(
    focal_lenghts_inv: torch.Tensor,
    img_shapes: torch.Tensor,
    kpt_xy_normalized: torch.Tensor,
    kpt_depth: torch.Tensor,
):
    n_imgs = focal_lenghts_inv.shape[0]
    focal_lenghts_inv_abs = torch.abs(focal_lenghts_inv)

    # 1) construct K_inv for each image in parallel and analytically
    # NOTE K_inv = [[1/f, 0, -px/f],[0, 1/f, -py/f],[0, 0, 1]]
    K_inv = torch.zeros((n_imgs, 3, 3), device=focal_lenghts_inv.device)
    K_inv[:, 0, 0] = focal_lenghts_inv_abs
    K_inv[:, 1, 1] = focal_lenghts_inv_abs
    K_inv[:, 2, 2] = 1
    img_max_size = torch.max(img_shapes, dim=1).values
    px = img_shapes[:, 1] / img_max_size / 2
    py = img_shapes[:, 0] / img_max_size / 2
    K_inv[:, 0, 2] = -px * focal_lenghts_inv_abs
    K_inv[:, 1, 2] = -py * focal_lenghts_inv_abs

    # 2) reproject kpt_xy to world coordinates
    kpt_xyz = torch.zeros(
        (n_imgs, kpt_xy_normalized.shape[1], 3), device=focal_lenghts_inv.device
    )
    kpt_xyz[:, :, 0] = kpt_xy_normalized[:, :, 0] * kpt_depth
    kpt_xyz[:, :, 1] = kpt_xy_normalized[:, :, 1] * kpt_depth
    kpt_xyz[:, :, 2] = kpt_depth
    kpt_xyz = kpt_xyz.unsqueeze(-1)  # (n_imgs, n_max_kpts, 3, 1)
    kpt_xyz = torch.matmul(K_inv.unsqueeze(1), kpt_xyz)  # (n_imgs, n_max_kpts, 3, 1)
    kpt_xyz = kpt_xyz.squeeze(-1)  # (n_imgs, n_max_kpts, 3)

    return kpt_xyz


def test_reproject_kpts(
    img_idx,
    xyz_train,
    query_xyz_list,
    img_shapes,
    query_xy_all_normalized_no_nan,
    depth_values_all_no_nan,
    query_xy_all_normalized,
):
    K = torch_utils.to_torch_tensor(
        geom_utils.get_calibration_matrix(*xyz_train[img_idx].shape[:2]), device="cpu"
    )
    query_xy = geom_utils.project_to_img_plane(query_xyz_list[img_idx], K=K)
    query_xy_unnormalised = query_xy * max(xyz_train[img_idx].shape)

    query_xyz_reconstructed = reproject_kpts(
        focal_lenghts_inv=torch.ones(img_shapes.shape[0]) / 0.3,
        img_shapes=img_shapes,
        kpt_xy_normalized=query_xy_all_normalized_no_nan,
        kpt_depth=depth_values_all_no_nan,
    )

    not_nan_mask = torch.logical_not(
        torch.any(torch.isnan(query_xy_all_normalized[img_idx]), dim=1)
    )
    err_reconstrunction = torch.mean(
        torch.norm(
            query_xyz_reconstructed[img_idx][not_nan_mask, :] - query_xyz_list[img_idx],
            dim=1,
        )
    )
    print("error", err_reconstrunction)


def focal_length_energy(
    focal_lenghts_inv: torch.Tensor,
    img_shapes: torch.Tensor,
    kpt_xy_normalized: torch.Tensor,
    kpt_depth: torch.Tensor,
    angles_mask: torch.Tensor,
    ratios_mask: torch.Tensor,
):
    """
    Args:
        focal_lenghts_inv: (n_imgs,) 1 / focal_length
        img_shapes: (n_imgs, 2). img_shapes[:,0] = img height (y), img_shapes[:,1] = img width (x)
        kpt_xy_normalized: (n_imgs, n_max_kpts, 2)
        kpt_depth: (n_imgs, n_max_kpts)
    """

    # 1) reproject keypoints to world coordinates
    kpt_xyz = reproject_kpts(
        focal_lenghts_inv=torch.abs(focal_lenghts_inv),
        img_shapes=img_shapes,
        kpt_xy_normalized=kpt_xy_normalized,
        kpt_depth=kpt_depth,
    )

    # 2) compute angles and ratios between all 3-tuples of keypoints for each image in parallel
    cos_angles, angles, ratios = (
        kpt_likelihood_utils.compute_angles_and_ratios_parallel(kpt_xyz)
    )
    cos_angles_mean, cos_angles_var, ratios_mean, ratios_var = (
        kpt_likelihood_utils.compute_stats_over_images(
            cos_angles, ratios, angles_mask, ratios_mask
        )
    )

    # 3) compute energy
    # valid_ratios = torch.logical_and(ratios > 0.01, ratios < 0.99)
    # valid_ratios = torch.logical_and(valid_ratios, ratios_mask)
    # ratios[~valid_ratios] = 0.5
    # min_ratios = torch.min(ratios, dim=1).values

    #  - torch.sum(torch.abs(cos_angles[angles_mask]))
    energy = torch.sum(cos_angles_var) + torch.sum(ratios_var) 
    # torch.mean(cos_angles_var) + torch.mean(ratios_var) 
    # energy = - 1000 * torch.mean(min_ratios)
    # TODO why does cos_angles*cos_anges work but cos_angles not work?
    # energy = torch.mean(torch.mean(cos_angles*cos_angles, dim=0))
    return (
        energy,
        cos_angles,
        ratios,
        cos_angles_mean,
        cos_angles_var,
        ratios_mean,
        ratios_var,
    )


def focal_length_energy_mean_dist(
    focal_lenghts_inv: torch.Tensor,
    img_shapes: torch.Tensor,
    kpt_xy_normalized: torch.Tensor,
    kpt_depth: torch.Tensor,
    angles_mask: torch.Tensor,
    ratios_mask: torch.Tensor,
    cos_angles_mean: torch.Tensor,
    ratios_mean: torch.Tensor,
):
    """
    Args:
        focal_lenghts_inv: (n_imgs,) 1 / focal_length
        img_shapes: (n_imgs, 2). img_shapes[:,0] = img height (y), img_shapes[:,1] = img width (x)
        kpt_xy_normalized: (n_imgs, n_max_kpts, 2)
        kpt_depth: (n_imgs, n_max_kpts)
    """

    # 1) reproject keypoints to world coordinates
    kpt_xyz = reproject_kpts(
        focal_lenghts_inv=torch.abs(focal_lenghts_inv),
        img_shapes=img_shapes,
        kpt_xy_normalized=kpt_xy_normalized,
        kpt_depth=kpt_depth,
    )

    # 2) compute angles and ratios between all 3-tuples of keypoints for each image in parallel
    cos_angles, _, ratios = (
        kpt_likelihood_utils.compute_angles_and_ratios_parallel(kpt_xyz)
    )
    cos_angles_dist2 = (cos_angles - cos_angles_mean) ** 2
    ratios_dist2 = (ratios - ratios_mean) ** 2
    energy = torch.sum(cos_angles_dist2[angles_mask] + ratios_dist2[ratios_mask])  #  cos_angles_dist2[angles_mask] + ratios_dist2[ratios_mask]

    return (
        energy,
        cos_angles,
        ratios
    )


def focal_length_energy_triangle_area(
    focal_lenghts_inv: torch.Tensor,
    img_shapes: torch.Tensor,
    kpt_xy_normalized: torch.Tensor,
    kpt_depth: torch.Tensor,
    triangle_mask: torch.Tensor,
):
    """
    Args:
        focal_lenghts_inv: (n_imgs,) 1 / focal_length
        img_shapes: (n_imgs, 2). img_shapes[:,0] = img height (y), img_shapes[:,1] = img width (x)
        kpt_xy_normalized: (n_imgs, n_max_kpts, 2)
        kpt_depth: (n_imgs, n_max_kpts)
    """

    # 1) reproject keypoints to world coordinates
    kpt_xyz = reproject_kpts(
        focal_lenghts_inv=torch.abs(focal_lenghts_inv) + 0.5,
        img_shapes=img_shapes,
        kpt_xy_normalized=kpt_xy_normalized,
        kpt_depth=kpt_depth,
    )

    # 2) compute squared triangle areas
    squared_trianlge_areas = kpt_likelihood_utils.compute_squared_triangle_areas_parallel(kpt_xyz)
    squared_trianlge_areas[~triangle_mask] = 0
    squared_trianlge_areas /= torch.max(squared_trianlge_areas, dim=1).values[:,None]
    squared_trianlge_areas[~triangle_mask] = torch.max(squared_trianlge_areas)
    squared_trianlge_areas_min = torch.min(squared_trianlge_areas, dim=1).values[:,None]

    # 3) compute energy
    # energy = -torch.sum(squared_trianlge_areas[triangle_mask])
    energy = -torch.sum(squared_trianlge_areas_min)

    return energy


def focal_length_energy_radius(
    focal_lenghts_inv: torch.Tensor,
    img_shapes: torch.Tensor,
    kpt_xy_normalized: torch.Tensor,
    kpt_depth: torch.Tensor,
    vertex_mask: torch.Tensor,
):
    # vertex_mask = torch.logical_not(torch.any(torch.isnan(kpt_xy_normalized), dim=-1))
    # 1) reproject keypoints to world coordinates
    kpt_xyz = reproject_kpts(
        focal_lenghts_inv=torch.abs(focal_lenghts_inv),
        img_shapes=img_shapes,
        kpt_xy_normalized=kpt_xy_normalized,
        kpt_depth=kpt_depth,
    )
    kpt_var, kpt_mean = torch_utils.masked_var_mean(kpt_xyz, vertex_mask[:,:,None], dim=1)

    # subtract mean
    kpt_xyz_centered = kpt_xyz - kpt_mean[:, None, :]
    kpt_xyz_centered[vertex_mask] = 0
    kpt_squared_radius = torch.sum(kpt_xyz_centered*kpt_xyz_centered, dim=2)  # (n_imgs, n_max_kpts)
    kpt_squared_radius = kpt_squared_radius / (torch.max(kpt_squared_radius, dim=1).values[:, None]+1e-6)
    _, kpt_squared_radius_mean = torch_utils.masked_var_mean(kpt_squared_radius, vertex_mask, dim=1)

    # maximize radius mean
    energy = -torch.sum(kpt_squared_radius_mean)
    return energy


def generate_is_not_nan_mask(x: torch.Tensor, dim=-1):
    """
    Args:
        x: (d1,d2,d3,...)
    """
    x_is_not_nan = torch.logical_not(torch.any(torch.isnan(x), dim=dim))
    return x_is_not_nan


def generate_angle_and_ratio_mask(kpts_is_not_nan: torch.Tensor):
    """
    Args:
        kpts_is_not_nan: (n_imgs, n_kpts) bool where True means that the kpt is not nan
    Returns:
        angles_mask: torch.Tensor of shape (n_imgs, n_kpts, n_kpts, n_kpts) where True means that the angle is not nan
        ratios_mask: torch.Tensor of shape (n_imgs, n_kpts, n_kpts, n_kpts) where True means that the ratio is not nan
    """

    pseudo_kpt_xyz = torch.zeros((kpts_is_not_nan.shape[0], kpts_is_not_nan.shape[1], 3))
    pseudo_kpt_xyz[~kpts_is_not_nan,:] = torch.nan
    pseudo_cos_angles, _, pseudo_ratios = kpt_likelihood_utils.compute_angles_and_ratios_parallel(pseudo_kpt_xyz)
    # triangle_areas = kpt_likelihood_utils.compute_squared_triangle_areas_parallel(pseudo_kpt_xyz)
    angles_mask = ~torch.isnan(pseudo_cos_angles)
    ratios_mask = ~torch.isnan(pseudo_ratios)
    # triangle_mask = ~torch.isnan(triangle_areas)
    return angles_mask, ratios_mask


# optimization loop
def update_focal_lengths(
    focal_lenghts_inv: torch.Tensor,
    energy_func: Callable,

    # params
    lr: float = 0.1,
    n_iter: int = 100,
):
    focal_lenghts_inv_moving = focal_lenghts_inv.clone().detach().requires_grad_(True)
    # optimizer = torch.optim.LBFGS([kpt_xyz], lr=lr)
    optimizer = torch.optim.Adam([focal_lenghts_inv_moving], lr=lr)

    for _ in tqdm(range(n_iter)):
        optimizer.zero_grad()
        total_energy = energy_func(focal_lenghts_inv_moving)
        total_energy.backward()
        optimizer.step()

    focal_lenghts_inv_res = focal_lenghts_inv_moving.clone().detach()
    return focal_lenghts_inv_res, total_energy


def optimize_focal_length_simple(
    kpt_xy_normalized: torch.Tensor,
    kpt_depth: torch.Tensor,
    kpt_is_not_nan: torch.Tensor,
    img_shapes: torch.Tensor,
    n_iter=1000,
    lr=0.001,
    focal_lengths_inv_opt: Optional[torch.Tensor] = None,
):
    """ Optimisation of focal lengths (similar results to simple_optimization)
    Args:
        kpt_xy_normalized: (n_imgs, n_max_kpts, 2)
        kpt_depth: (n_imgs, n_max_kpts)
        kpt_is_not_nan: (n_imgs, n_max_kpts) bool tensor where True means that the kpt is not nan
        img_shapes: (n_imgs, 2). img_shapes[:,0] = img height (y), img_shapes[:,1] = img width (x)
    """
    angles_mask, ratios_mask = generate_angle_and_ratio_mask(kpt_is_not_nan)

    # define energy function
    def energy_func(focal_lenghts_inv):
        return focal_length_energy(
            focal_lenghts_inv=focal_lenghts_inv,
            img_shapes=img_shapes,
            kpt_xy_normalized=kpt_xy_normalized,
            kpt_depth=kpt_depth,
            angles_mask=angles_mask,
            ratios_mask=ratios_mask,
        )[0]
    
    # run optimization loop
    if focal_lengths_inv_opt is None:
        focal_lengths_inv_opt = torch.ones(img_shapes.shape[0])/5
    focal_lengths_inv_opt, total_energy = update_focal_lengths(
        focal_lenghts_inv=focal_lengths_inv_opt,
        energy_func=energy_func,
        lr=lr,
        n_iter=n_iter
    )
    print(total_energy, focal_lengths_inv_opt, sep="\n")
    return focal_lengths_inv_opt


def optimize_focal_length_global_local(
    kpt_xy_normalized: torch.Tensor,
    kpt_depth: torch.Tensor,
    kpt_is_not_nan: torch.Tensor,
    img_shapes: torch.Tensor,
    n_global_iter = 10,
    n_local_iter = 100,
    lr = 0.001,
    focal_lengths_inv_opt: Optional[torch.Tensor] = None,
):
    """ Global local optimisation of focal lengths (similar results to simple optimization)
    Args:
        kpt_xy_normalized: (n_imgs, n_max_kpts, 2)
        kpt_depth: (n_imgs, n_max_kpts)
        kpt_is_not_nan: (n_imgs, n_max_kpts) bool tensor where True means that the kpt is not nan
        img_shapes: (n_imgs, 2). img_shapes[:,0] = img height (y), img_shapes[:,1] = img width (x)
    """
    if focal_lengths_inv_opt is None:
        focal_lengths_inv_opt = torch.ones(img_shapes.shape[0])/5

    # compute angle and ratio mask
    angles_mask, ratios_mask = generate_angle_and_ratio_mask(kpt_is_not_nan)

    n_global_iter = 10
    for _ in range(n_global_iter):
        # 1) reproject keypoints to world coordinates
        kpt_xyz = reproject_kpts(
            focal_lenghts_inv=torch.abs(focal_lengths_inv_opt),
            img_shapes=img_shapes,
            kpt_xy_normalized=kpt_xy_normalized,
            kpt_depth=kpt_depth,
        )

        # 2) compute current cos_angle and ratio mean
        cos_angles, _, ratios = (
                kpt_likelihood_utils.compute_angles_and_ratios_parallel(kpt_xyz)
        )
        cos_angles_mean, _, ratios_mean, _ = (
            kpt_likelihood_utils.compute_stats_over_images(
                cos_angles, ratios, angles_mask, ratios_mask
            )
        )

        # 3) define energy function with current mean values
        def energy_func(focal_lenghts_inv):
            return focal_length_energy_mean_dist(
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
        focal_lengths_inv_opt, total_energy = update_focal_lengths(
            focal_lenghts_inv=focal_lengths_inv_opt,
            energy_func=energy_func,
            lr=lr,
            n_iter=n_local_iter
        )
        # focal_lengths_inv_opt_cur = torch.clip(focal_lengths_inv_opt_cur, 0.1, 1.0)
        print(total_energy, focal_lengths_inv_opt, sep="\n")
    
    return focal_lengths_inv_opt
