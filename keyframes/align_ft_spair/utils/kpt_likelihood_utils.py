from typing import Optional
import torch
import numpy as np
from keyframes.align_ft_spair.utils import ft_align_utils, spair_utils, geom_utils
from utils import torch_utils
from tqdm import tqdm


def load_airplane_kpt_xyz():
    point_coords = {
        0: [203.949997, 18.889400, 149.082993], #"nose",
        1: [204.106995, 27.134600, 141.526993], #"cockpit",
        2: [203.949997, 31.383801, 134.893997], #"forehead",
        3: [202.848007, 1.248820, 136.585007], #"landing_gear_front",
        # -- right / left --
        #4: "landing_gear_right",
        5: [212.306000, 0.262179, 55.779202], #"landing_gear_left",
        #6: "engine_front_right",
        7: [264.902008, 23.557501, 83.425201], #"engine_front_left",
        #8: "wing_end_right",
        9: [323.007996, 36.138699, 14.594200], #"wing_end_left",
        #10: "engine_back_right",
        11: [272.032990, 19.853201, 60.315701],# "engine_back_left",
        #12: "wing_foot_front_right",
        13: [219.091995, 34.928101, 80.031799], #"wing_foot_front_left", 
        #14: "wing_foot_back_right",
        15: [221.417999, 29.979200, 36.881802], #"wing_foot_back_left",
        #16: "tailplane_end_right", # tailplane = horizontal stabilizer
        17: [249.998001, 72.005997, -88.522903], #"tailplane_end_left",
        #18: "tailplane_foot_front_right",
        19: [206.414001, 74.796204, -62.024899], #"tailplane_foot_front_left",
        #20: "tailplane_foot_back_right",
        21: [206.414001, 72.803101, -82.024902], #"tailplane_foot_back_left",
        # ---------------
        22: [203.949005, 37.227600, -19.331900], #"stabilizer_vertical_foot",
        23: [204.531006, 77.370201, -91.800201], #"stabilizer_vertical_end",
        24: [203.598999, 32.620899, -74.676201], #"rear", # can also be rear engine
    }
    center_x = point_coords[0][0]
    for idx in [4,6,8,10,12,14,16,18,20]:
        diff_x = point_coords[idx+1][0] - center_x
        point_coords[idx] = [center_x - diff_x, point_coords[idx+1][1], point_coords[idx+1][2]]
    
    kpt_xyz = np.zeros((25,3), dtype=float)
    for idx in range(25):
        kpt_xyz[idx][:] = point_coords[idx]

    return kpt_xyz


def compute_angles_and_ratios(
    kpts_xyz: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    """
    Args:
        - kpts_xyz: (N, 3) tensor
    """
    edges = kpts_xyz[None,:,:] - kpts_xyz[:,None,:]  # (N, N, 3)
    edge_norms = torch.norm(edges, dim=-1)  # (N, N)
    dots =  torch.sum(edges[:,:,None,:] * edges[:,None,:,:], dim=-1)  # (N, N, N)
    norm_prods = edge_norms[:,:,None] * edge_norms[:,None,:]  # (N, N, N)
    # NOTE angles[i,i,i] and ratios[i,i,i] is not well defined but that does not matter
    # (angle for i,j,k where |{i,j,k}| <= 2 is also not relevant)
    angles = torch.acos(dots / (norm_prods + 1e-6))  # (N, N, N)
    ratios = edge_norms[:,:,None] / (edge_norms[:,:,None] + edge_norms[:,None,:] + 1e-6)  # (N, N, N)
    return angles, ratios # , edges, edge_norms, dots, norm_prods


def compute_angles_and_ratios_parallel(kpts_xyz: torch.Tensor):
    """
    Computes angles and ratios between keypoints in 3D space across multiple images in parallel.

    This function calculates the pairwise vector differences between keypoints for each image,
    then computes the angles and ratios based on these differences. The computation is vectorized
    and operates on all images in parallel for efficiency.

    Args:
        kpts_xyz (torch.Tensor): A tensor of shape (n_imgs, N, 3) containing the 3D coordinates
                                 of keypoints for each image. `n_imgs` is the number of images,
                                 and `N` is the number of keypoints per image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - Angles: A tensor of shape (n_imgs, N, N, N) representing the angles between each
                      triplet of keypoints for each image.
            - Ratios: A tensor of shape (n_imgs, N, N, N) representing the ratios of distances
                      between each pair of keypoints for each image.
    """
    # Step 1: Compute Edges
    edges = kpts_xyz[:, None, :, :] - kpts_xyz[:, :, None, :]  # (n_imgs, N, N, 3)
    
    # Step 2: Compute Edge Norms
    edge_norms = torch.norm(edges, dim=-1)  # (n_imgs, N, N)
    
    # Step 3: Compute Dot Products
    dots = torch.sum(edges[:, :, :, None, :] * edges[:, :, None, :, :], dim=-1)  # (n_imgs, N, N, N)
    
    # Step 4: Compute Norm Products
    norm_prods = edge_norms[:, :, :, None] * edge_norms[:, :, None, :]  # (n_imgs, N, N, N)
    
    # Step 5: Compute Angles
    # TODO do not consider entries where i=j=k
    cos_angles = dots / (norm_prods + 1e-6)  # (n_imgs, N, N, N)
    angles = torch.acos(cos_angles)  # (n_imgs, N, N, N)
    
    # Step 6: Compute Ratios
    ratios = edge_norms[:, :, :, None] / (edge_norms[:, :, :, None] + edge_norms[:, :, None, :] + 1e-6)  # (n_imgs, N, N, N)
    
    # Step 7: Return
    return cos_angles, angles, ratios


# TODO rename this function
def compute_stats_over_images(
    angles: torch.Tensor, 
    ratios: torch.Tensor,
    angles_mask: torch.Tensor,
    ratios_mask: torch.Tensor
):
    """
    Args:
        angles (torch.Tensor): A tensor of shape (n_imgs, N, N, N) representing the angles between
                               each triplet of keypoints for each image. `n_imgs` is the number of images,
                               and `N` is the number of keypoints per image.
        ratios (torch.Tensor): A tensor of shape (n_imgs, N, N, N) representing the ratios of distances
                               between each pair of keypoints for each image.
        angles_mask (torch.Tensor): A tensor of shape (n_imgs, N, N, N) containing boolean values
                                    indicating which angles are valid.
        ratios_mask (torch.Tensor): A tensor of shape (n_imgs, N, N, N) containing boolean values
                                    indicating which ratios are valid.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing four tensors:
            - Angles Mean: The mean of angles across all images.
            - Angles Variance: The variance of angles across all images.
            - Ratios Mean: The mean of ratios across all images.
            - Ratios Variance: The variance of ratios across all images.
    """
    # Compute Mean and SD for Angles
    angles_var, angles_mean = torch_utils.masked_var_mean(angles, angles_mask)
    # angles_var, angles_mean = torch.var_mean(angles[angles_mask], dim=0)
    # angles_mean = torch.nanmean(angles, dim=0)
    # angles_var = torch_utils.nanvar(angles, dim=0)
    
    # Compute Mean and SD for Ratios
    ratios_var, ratios_mean = torch_utils.masked_var_mean(ratios, ratios_mask)
    # ratios_var, ratios_mean = torch.var_mean(ratios[ratios_mask], dim=0)
    # ratios_mean = torch.nanmean(ratios, dim=0)
    # ratios_var = torch_utils.nanvar(ratios, dim=0)
    
    return angles_mean, angles_var, ratios_mean, ratios_var


def compute_angles_and_ratios_for_query_xyz(
    kpts_xyz: torch.Tensor,
    query_xyz: torch.Tensor,
):
    """
    Args:
        - kpts_xyz: (K, 3) tensor
        - query_xyz: (N, 3) tensor
    Returns:
        - angles: (N, K, K) tensor
        - ratios: (N, K, K) tensor
    """
    q_kpt_edges =  kpts_xyz[None,:,:] - query_xyz[:,None,:]  # (N, K, 3)
    kpt_kpt_edges = kpts_xyz[None,:,:] - kpts_xyz[:,None,:]  # (K, K, 3)
    q_kpt_edge_norms = torch.norm(q_kpt_edges, dim=-1)  # (N, K)
    kpt_kpt_edge_norms = torch.norm(kpt_kpt_edges, dim=-1)  # (K, K)
    dots = torch.sum(q_kpt_edges[:,:,None,:] * kpt_kpt_edges[None,:,:,:], dim=-1) # (N, K, K)
    norm_prods = q_kpt_edge_norms[:,:,None] * kpt_kpt_edge_norms[None,:,:]  # (N, K, K)
    angles = torch.acos(dots / (norm_prods + 1e-6))  # (N, K, K)
    ratios = q_kpt_edge_norms[:,:,None] / (q_kpt_edge_norms[:,:,None] + kpt_kpt_edge_norms[None,:,:] + 1e-6)  # (N, K, K)
    return angles, ratios


def compute_signed_volumes(vertices):
    """
    Compute the signed volumes of all tetrahedrons formed by combinations of four vertices.
    
    Parameters:
    vertices (torch.Tensor): Tensor of shape (N, 3) representing the coordinates of N vertices.
    
    Returns:
    torch.Tensor: Tensor containing the signed volumes of all tetrahedrons.
    """
    # Generate all combinations of four vertices
    N = vertices.shape[0]
    combinations = torch.combinations(torch.arange(N), r=4)
    
    # Extract the vertex coordinates for each combination
    tetrahedrons = vertices[combinations]  # shape: (num_combinations, 4, 3)

    # Scale tetrahedrons by average edge length
    avg_edge_length = torch.mean(torch.norm(tetrahedrons[:, None, :, :] - tetrahedrons[:, :, None, :], dim=-1))
    tetrahedrons = tetrahedrons / avg_edge_length
    
    # Create the 4x4 matrices
    ones = torch.ones((tetrahedrons.shape[0], 4, 1), device=vertices.device)
    matrices = torch.cat((tetrahedrons, ones), dim=2)
    
    # Compute the determinants
    determinants = torch.det(matrices)
    
    # Calculate the signed volumes
    signed_volumes = determinants / 6
    
    return signed_volumes


def get_compute_signed_volumes_parallel_fn(
    # vertices: torch.Tensor,
    batch_size: int,
    n_vertices_per_batch: int,
    device: str,
    masked_combinations: Optional[torch.Tensor] = None
):
    n_imgs, N = batch_size, n_vertices_per_batch
    combinations = torch.combinations(torch.arange(N), r=4)
    if masked_combinations is not None:
        combinations = combinations[masked_combinations]
    num_combinations = combinations.shape[0]
    
    # Repeat combinations for each image and adjust indices for batched access
    batch_indices = torch.arange(n_imgs).view(-1, 1, 1).repeat(1, num_combinations, 4)
    combinations_expanded = combinations.repeat(n_imgs, 1, 1)

    ones = torch.ones((n_imgs, num_combinations, 4, 1), device=device)

    # transfer to device
    batch_indices = batch_indices.to(device)
    combinations_expanded = combinations_expanded.to(device)
    ones = ones.to(device)

    def compute_signed_volumes_parallel_(vertices):
        return compute_signed_volumes_parallel(
            vertices=vertices, 
            batch_indices=batch_indices, 
            combinations_expanded=combinations_expanded, 
            ones=ones
        )
    
    return compute_signed_volumes_parallel_, batch_indices, combinations_expanded, ones


def compute_signed_volumes_parallel(
    vertices: torch.Tensor,
    batch_indices: Optional[torch.Tensor] = None,
    combinations_expanded: Optional[torch.Tensor] = None,
    ones: Optional[torch.Tensor] = None,
):
    """
    Compute the signed volumes of all tetrahedrons formed by combinations of four vertices for multiple images.
    
    Parameters:
        vertices: (n_imgs, N, 3) representing the coordinates of vertices for n_imgs images.
        batch_indices: (n_imgs, num_combinations, 4) containing the indices of the vertices for each image.
        combinations_expanded: (n_imgs, num_combinations, 4) containing the combinations of vertices for each image.
        ones: (n_imgs, num_combinations, 4, 1) tensor containing ones for each image.
    
    Returns:
    torch.Tensor: Tensor containing the signed volumes of all tetrahedrons for each image.
    """
    n_imgs, N, _ = vertices.shape

    if batch_indices is None or ones is None or combinations_expanded is None:
        combinations = torch.combinations(torch.arange(N), r=4)
        num_combinations = combinations.shape[0]
        
        # Repeat combinations for each image and adjust indices for batched access
        batch_indices = torch.arange(n_imgs).view(-1, 1, 1).repeat(1, num_combinations, 4)
        combinations_expanded = combinations.repeat(n_imgs, 1, 1)

        ones = torch.ones((n_imgs, num_combinations, 4, 1), device=vertices.device)    
    
    # Extract the vertex coordinates for each combination for all images
    tetrahedrons = vertices[batch_indices, combinations_expanded]  # shape: (n_imgs, num_combinations, 4, 3)

    # Compute average edge length for each image
    edge_lengths = torch.norm(tetrahedrons[:, :, None, :, :] - tetrahedrons[:, :, :, None, :], dim=-1)
    avg_edge_length = torch.mean(edge_lengths, dim=(2, 3))
    
    # Normalize tetrahedrons by the average edge length for each image
    tetrahedrons_normalized = tetrahedrons / (avg_edge_length[:,:,None,None] + 1e-6)
    
    # Create the 4x4 matrices for each tetrahedron in each image
    print("devices", tetrahedrons_normalized.device, ones.device)
    matrices = torch.cat((tetrahedrons_normalized, ones), dim=3)  # shape: (n_imgs, num_combinations, 4, 4)
    
    # Compute the determinants for each image
    determinants = torch.det(matrices)
    
    # Calculate the signed volumes for each image
    signed_volumes = determinants / 6
    
    return signed_volumes


def test_compute_signed_volumes():
    vertices = torch.tensor([
        [1, 1, 1],
        [2, 3, 1],
        [4, 1, 2],
        [1, 2, 3],
        [0, 0, 0]
    ], dtype=torch.float)
    
    signed_volumes = compute_signed_volumes(vertices)
    print(signed_volumes)
    assert torch.allclose(signed_volumes, torch.tensor([1.6667, 1.6667, 1.6667, -1.6667, 0.0000]))


def compute_signed_volumes_query(vertices, query_xyz):
    """
    Compute the signed volumes of all tetrahedrons formed by combinations of three vertices from 'vertices'
    and one vertex from 'query_xyz'.
    
    Parameters:
    vertices (torch.Tensor): Tensor of shape (N, 3) representing the coordinates of N vertices.
    query_xyz (torch.Tensor): Tensor of shape (Q, 3) representing the coordinates of Q query vertices.
    
    Returns:
    torch.Tensor: Tensor containing the signed volumes of all tetrahedrons.
    """
    N = vertices.shape[0]
    Q = query_xyz.shape[0]
    
    # Generate all combinations of three vertices from 'vertices'
    combinations = torch.combinations(torch.arange(N), r=3)
    
    # Extract the vertex coordinates for each combination
    base_tetrahedrons = vertices[combinations]  # shape: (num_combinations, 3, 3)
    
    # Repeat the query vertices to match the number of combinations
    repeated_query = query_xyz.unsqueeze(1).repeat(1, base_tetrahedrons.shape[0], 1, 1)  # shape: (Q, num_combinations, 1, 3)
    
    # Repeat the base tetrahedrons to match the number of query vertices
    repeated_base_tetrahedrons = base_tetrahedrons.unsqueeze(0).repeat(Q, 1, 1, 1)  # shape: (Q, num_combinations, 3, 3)
    
    # Combine each query vertex with each base tetrahedron
    tetrahedrons = torch.cat((repeated_base_tetrahedrons, repeated_query), dim=2)  # shape: (Q, num_combinations, 4, 3)
    
    # Create the 4x4 matrices
    ones = torch.ones((tetrahedrons.shape[0], tetrahedrons.shape[1], 4, 1), device=vertices.device)
    matrices = torch.cat((tetrahedrons, ones), dim=3)
    
    # Compute the determinants
    determinants = torch.det(matrices)
    
    # Calculate the signed volumes
    signed_volumes = determinants / 6
    
    return signed_volumes


def test_compute_signed_volumes_query():
    vertices = torch.tensor([
        [1, 1, 1],
        [2, 3, 1],
        [4, 1, 2],
        [1, 2, 3],
        [0, 0, 0]
    ], dtype=torch.float)
    
    query_xyz = torch.tensor([
        [1, 1, 0],
        [2, 2, 2]
    ], dtype=torch.float)
    
    signed_volumes = compute_signed_volumes_query(vertices, query_xyz)
    print(signed_volumes)
    assert torch.allclose(signed_volumes, torch.tensor([
        [1.6667, 1.6667, 1.6667, -1.6667, 0.0000],
        [1.0000, 1.0000, 1.0000, -1.0000, 0.0000]
    ]))



def compute_squared_triangle_areas_parallel(vertices):
    """
    Compute the areas of all triangles formed by combinations of three vertices for each image in parallel.
    
    Parameters:
    vertices (torch.Tensor): Tensor of shape (n_imgs, k, 3) representing the coordinates of k vertices in each of n_imgs images.
    
    Returns:
    torch.Tensor: Tensor containing the areas of all triangles for each image, in parallel.
    """
    n_imgs, k, _ = vertices.shape
    combinations = torch.combinations(torch.arange(k), r=3)
    # Expand vertices to match the number of combinations
    vertices_expanded = vertices[:, combinations]
    
    # Calculate the lengths of the sides of each triangle using broadcasting
    a = torch.norm(vertices_expanded[:, :, 1] - vertices_expanded[:, :, 0], dim=2)
    b = torch.norm(vertices_expanded[:, :, 2] - vertices_expanded[:, :, 1], dim=2)
    c = torch.norm(vertices_expanded[:, :, 0] - vertices_expanded[:, :, 2], dim=2)
    
    # Calculate the semiperimeter
    s = (a + b + c) / 2
    
    # Calculate the area of each triangle using Heron's formula
    areas = s * (s - a) * (s - b) * (s - c)
    
    return areas


def compute_new_props():
    # TODO
    pass


def compute_alignment_energy():
    """ energy based on alignment between model and reference
    """
    pass


def compute_model_energy():
    """ internal energy of model
    """
    pass

def compute_kpt_likelihood_from_kpt_attn(
    kpt_attn: torch.Tensor, 
    kpt_attn_mean: torch.Tensor,
    kpt_attn_sd: torch.Tensor,
):
    """ Compute likelihood of attn value being in range [-infty, -x] and [x, infty],
        where x = |attn - mean| and the normal distribution is defined by mean=0 and sd.
        NOTE: at x=0, this value is 1.
    Args:
        - kpt_attn: (K, H, W) tensor
        - kpt_attn_mean: (K,) tensor
        - kpt_attn_sd: (K,) tensor
    Returns:
        - kpt_likelihood: (K, H, W) tensor
    """
    kpt_attn_mean = kpt_attn_mean[:,None,None]
    kpt_normal_distribution = torch.distributions.Normal(
        kpt_attn_mean, 
        kpt_attn_sd[:,None,None]
    )
    kpt_cdf = kpt_normal_distribution.cdf(kpt_attn_mean + torch.abs(kpt_attn - kpt_attn_mean))
    kpt_likelihood = 2 * (1 - kpt_cdf)
    return kpt_likelihood


def compute_kpt_label_likelihood(
    img_embd_hat: torch.Tensor,
    kpt_embd_hat_mean: torch.Tensor,
    kpt_attn_mean: torch.Tensor,
    kpt_attn_sd: torch.Tensor,
):
    """
    Args:
        - img_embd_hat: (C, H, W)
        - kpt_embd_hat_mean: (K,C)
        - kpt_attn_mean: (K,)
        - kpt_attn_sd: (K,)
    """
    # 1) compute attn between img_embd and kpt_embd
    kpt_attn = ft_align_utils.compute_attn(
        kpt_embd_hat_mean,
        img_embd_hat[None,:,:,:],
        enable_normalize=False,
        enable_softmax=False
    )  # (K, H, W)

    # 2) compute keypoint likelihood
    kpt_likelihood = compute_kpt_likelihood_from_kpt_attn(
        kpt_attn, 
        kpt_attn_mean, 
        kpt_attn_sd
    )  # (K, H, W)

    return kpt_likelihood


def get_downsampled_img_xyz(
    img_idx: int,
    embd_size: int,
    xyz_train_np: np.ndarray,
):
    # extract depth
    h, w, _ = xyz_train_np[img_idx].shape
    img_coords = []
    for i in range(60):
        for j in range(60):
            x, y = spair_utils.transform_image_coords_inv(
                j+0.5, i+0.5, w, h, embd_size, pad=True
            )
            if x < 0 or y < 0 or x >= w or y >= h:
                x, y = 0,0
            img_coords.append([x, y])
    img_coords = np.array(img_coords).astype(int)
    xyz_train_downsampled = xyz_train_np[img_idx][img_coords[:,1], img_coords[:,0], :]
    return img_coords, xyz_train_downsampled


# compute other input args
def compute_img_xyz_and_K(
    img_idx: int,
    xyz_train_np: np.ndarray,
    embd_size: int,
):
    """
    Args:
        - img_idx: int
        - xyz_train: (N, H, W, 3)
    """
    img_coords, xyz_train_downsampled = get_downsampled_img_xyz(
        img_idx, embd_size, xyz_train_np
    )
    img_h, img_w = xyz_train_np[img_idx].shape[:2]
    K = geom_utils.get_calibration_matrix(img_h, img_w)

    # to torch
    img_xyz = torch_utils.to_torch(xyz_train_downsampled, device="cpu")
    K = torch_utils.to_torch(K, device="cpu")
    img_xy = torch_utils.to_torch(img_coords / max(img_h, img_w), device="cpu")
    return img_xy, img_xyz, K


# ========================
# Energies
# ========================


def my_super_energy(
    img_xyz: torch.Tensor,
    img_embd_hat: torch.Tensor,
    kpt_xyz: torch.Tensor,
    kpt_embd_hat_mean: torch.Tensor,
    kpt_attn_mean: torch.Tensor,
    kpt_attn_sd: torch.Tensor,
    kpt_ratios: torch.Tensor,
    kpt_angle: torch.Tensor  
):
    """
    Args:
        - img_xyz: (H*W, 3)
        - img_embd_hat: (C, H, W)
        - kpt_xyz: (K, 3)
        - kpt_embd_hat_mean: (K, C)
        - kpt_attn_mean: (K,)
        - kpt_attn_sd: (K,)
        - kpt_ratios: (K**3, 1)
        - kpt_angle: (K**3, 1)
    """
    c, h, w = img_embd_hat.shape

    # TODO this can be pre-computed
    # 1) compute attn between img_embd and kpt_embd
    kpt_attn = ft_align_utils.compute_attn(
        kpt_embd_hat_mean,
        img_embd_hat[None,:,:,:],
        enable_normalize=False,
        enable_softmax=False
    )  # (K, H, W)

    # 2) compute keypoint likelihood
    kpt_likelihood = compute_kpt_label_likelihood(
        kpt_attn, 
        kpt_attn_mean, 
        kpt_attn_sd
    )  # (K, H, W)

    # 3) compute dists between kpt_xyz and img_xyz
    # NOTE use p=1 (or a mix of p=1,2) to address problem of multiple keypoint matches
    # or take minimum among all keypoint matches
    dists = torch.cdist(kpt_xyz, img_xyz, p=1)  # (K, N) = (K,H*W)

    # 4) compute alignment energy
    alignment_energy = torch.sum(dists * kpt_likelihood.reshape((-1, h*w)))

    # 3) compute model energy
    cur_angles, cur_ratios = compute_angles_and_ratios(kpt_xyz)
    model_angle_energy = torch.sum((cur_angles - kpt_angle)**2)
    # model_ratio_energy = torch.sum((cur_ratios - kpt_ratios)**2)
    # model_energy = model_angle_energy + model_ratio_energy
    model_energy = model_angle_energy

    # 4) total energy
    total_energy = alignment_energy + model_energy
    return total_energy


def my_super_energy_v2(
    img_xyz: torch.Tensor,
    kpt_likelihood: torch.Tensor,
    kpt_xyz: torch.Tensor,
    kpt_ratios: torch.Tensor,
    kpt_angle: torch.Tensor  
):
    """
    Args:
        - img_xyz: (H*W, 3)
        - kpt_likelihood: (K, H, W)
        - kpt_xyz: (K, 3)
        - kpt_ratios: (K**3, 1)
        - kpt_angle: (K**3, 1)
    """
    k, h, w = kpt_likelihood.shape
    # k, n = dists.shape
    n = h*w

    # 1) compute dists between kpt_xyz and img_xyz
    # NOTE use p=1 (or a mix of p=1,2) to address problem of multiple keypoint matches
    # or take minimum among all keypoint matches
    # TODO adjust
    # dists = torch.cdist(kpt_xyz, img_xyz, p=2)  # (K, N) = (K,H*W)
    # dists = dists / (dists + 0.1)
    # dists_min = torch.min(dists / (kpt_likelihood.reshape((-1, h*w)) + 1e-8), dim=-1)[0]
    # gravity based:
    dists = torch.cdist(kpt_xyz, img_xyz, p=1)  # (K, N) = (K,H*W)
    dists = - 10 / (dists + 1e-8)

    # 2) compute alignment energy
    alignment_energy = torch.sum(dists * kpt_likelihood.reshape((-1, n)))
    # alignment_energy = torch.sum(dists_min)

    # 3) compute model energy
    cur_angles, cur_ratios = compute_angles_and_ratios(kpt_xyz)
    model_angle_energy = torch.sum((cur_angles - kpt_angle)**2)
    model_ratio_energy = torch.sum((cur_ratios - kpt_ratios)**2)
    model_energy = model_angle_energy + model_ratio_energy

    # 4) total energy
    total_energy = alignment_energy  + 0.1*model_energy
    return total_energy


def my_super_energy_v3(
    img_xyz: torch.Tensor,
    kpt_likelihood: torch.Tensor,
    kpt_xyz: torch.Tensor,
    kpt_ratios: torch.Tensor,
    kpt_angle: torch.Tensor,
    K: torch.Tensor,
    img_xy: torch.Tensor,
    model_energy_weight: float = 0.01
):
    """ Minimize alignment not in 3D but on the 2D image plane
    Args:
        - img_xyz: (H*W, 3)
        - kpt_likelihood: (K, H, W)
        - kpt_xyz: (K, 3)
        - kpt_ratios: (K**3, 1)
        - kpt_angle: (K**3, 1)
        - K: (3,3) camera intrinsic matrix (camera transformation matric is assumed to be identity)
        - img_xy: (H*W,2) xy coords of image pixels, normalized by max(img_w, img_h)
    """
    k, h, w = kpt_likelihood.shape
    # k, n = dists.shape
    n = h*w

    kpt_xy_harmonic = kpt_xyz @ K[:3,:3].T
    kpt_xy = kpt_xy_harmonic[:,:2] / kpt_xy_harmonic[:,2][:,None]

    # 1) compute dists between kpt_xyz and img_xyz
    # NOTE use p=1 (or a mix of p=1,2) to address problem of multiple keypoint matches
    # or take minimum among all keypoint matches
    # TODO adjust
    dists3d = torch.cdist(kpt_xyz, img_xyz, p=2)  # (K, N) = (K,H*W)
    # dists = dists / (dists + 0.1)
    # dists_min = torch.min(dists / (kpt_likelihood.reshape((-1, h*w)) + 1e-8), dim=-1)[0]
    # gravity based:
    # dists = torch.cdist(kpt_xyz, img_xyz, p=1)  # (K, N) = (K,H*W)
    # dists = - 10 / (dists + 1e-8)
    dists_on_img_plane = torch.cdist(kpt_xy, img_xy, p=2)  # (K,H*W)
    dists_on_img_plane_min = torch.min(dists_on_img_plane / (kpt_likelihood.reshape((-1, h*w)) + 1e-8), dim=-1)[0]

    # 2) compute alignment energy
    alignment_energy3d = torch.sum(dists3d * kpt_likelihood.reshape((-1, n)))
    # alignment_energy = torch.sum(dists_min)
    alignment_energy2d = torch.sum(dists_on_img_plane_min)
    alignment_energy = alignment_energy2d + alignment_energy3d

    # 3) compute model energy
    cur_angles, cur_ratios = compute_angles_and_ratios(kpt_xyz)
    model_angle_energy = torch.sum((cur_angles - kpt_angle)**2)
    model_ratio_energy = torch.sum((cur_ratios - kpt_ratios)**2)
    model_energy = model_angle_energy  # + model_ratio_energy

    # 4) total energy
    total_energy = alignment_energy  + model_energy_weight*model_energy
    return total_energy



# ========================
# Optimization
# ========================


# let's choose two keypoints and look at the energy
def eval_energy_for_chosen(kpt_xyz_cur, chosen_mask, kpt_xyz_ref):
    # TODO we are only interested in angles between query xyz in img_xyz and the
    # chose kpt xyz
    ref_angles, ref_ratios = compute_angles_and_ratios_for_query_xyz(
        kpt_xyz_ref[chosen_mask,:],
        kpt_xyz_ref[chosen_mask,:]
    )  # (num_chosen, num_chosen, num_chosen)
    num_chosen = kpt_xyz_ref[chosen_mask,:].shape[0]
    ref_angles = ref_angles.reshape((num_chosen, -1))
    ref_ratios = ref_ratios.reshape((num_chosen, -1))
    cur_angles, cur_ratios = compute_angles_and_ratios_for_query_xyz(
        kpt_xyz_cur[chosen_mask,:],
        kpt_xyz_cur[chosen_mask,:],
    )  # (num_chosen, num_chosen, num_chosen)
    cur_angles = cur_angles.reshape((num_chosen, -1))
    cur_ratios = cur_ratios.reshape((num_chosen, -1))

    model_angle_energy = torch.sum((ref_angles - cur_angles)**2, dim=-1)
    model_ratio_energy = torch.sum((ref_ratios - cur_ratios)**2, dim=-1)
    model_energy = model_angle_energy + model_ratio_energy  # (num_chosen,)
    return torch.sum(model_energy)


def compute_repulsive_energy_exp_factor(r, a):
    # NOTE if energy is supposed to be alpha at r, then exp_factor = 1 / (log(1+a)*r**2) 
    # becuase => e^(log(1+a)) = alpha + 1 => a = alpha
    return 1/(np.log(1+a)*r**2)


def compute_repulsive_energy(kpt_xyz_cur, chosen_mask, img_xyz, exp_factor=1000.0):
    """ compute repulsive energy so that two kpt do not land at the same spot
    Args:
        - kpt_xyz_cur: (K, 3)
        - chosen_mask: (K,)
        - img_xyz: (h*w, 3)
    """
    dists_to_chosen = torch.cdist(kpt_xyz_cur[chosen_mask], img_xyz, p=2)  # (K, N)
    not_chosen = torch.logical_not(chosen_mask)
    energy = torch.sum(torch.exp(1/(exp_factor * dists_to_chosen)) - 1, dim=0)  # (N,)
    repulsive_energy = torch.zeros((kpt_xyz_cur.shape[0], img_xyz.shape[0]), dtype=energy.dtype)
    repulsive_energy[not_chosen, :] = energy  # energy is the same for all not chosen keypoints
    return repulsive_energy


def eval_new_energy():
    """
    """
    
    pass


# let's choose two keypoints and look at the energy
def eval_energy_for_query_xyz(kpt_xyz_cur, chosen_mask, img_xyz, kpt_xyz_ref, repulsive_weight=0.5, exp_factor=1000.0):
    # TODO add repulsive energy so that two kpt do not land at the same spot
    # TODO we are only interested in angles between query xyz in img_xyz and the
    # chose kpt xyz
    non_chosen = torch.logical_not(chosen_mask)
    ref_angles, ref_ratios = compute_angles_and_ratios_for_query_xyz(
        kpt_xyz_ref[chosen_mask,:],
        kpt_xyz_ref[non_chosen,:]
    )  # (K - num_chosen, num_chosen, num_chosen)
    K = kpt_xyz_ref.shape[0]
    num_chosen = kpt_xyz_ref[chosen_mask,:].shape[0]
    ref_angles = ref_angles.reshape((K - num_chosen, -1))
    ref_ratios = ref_ratios.reshape((K - num_chosen, -1))
    cur_angles, cur_ratios = compute_angles_and_ratios_for_query_xyz(
        kpt_xyz_cur[chosen_mask,:],
        img_xyz,
    )  # (h*w, nc, nc)
    n = img_xyz.shape[0]
    cur_angles = cur_angles.reshape((n, -1))
    cur_ratios = cur_ratios.reshape((n, -1))

    model_angle_energy = torch.sum((ref_angles[:,None,:] - cur_angles[None,:,:])**2, dim=-1)
    model_ratio_energy = torch.sum((ref_ratios[:,None,:] - cur_ratios[None,:,:])**2, dim=-1)
    model_energy = model_angle_energy + model_ratio_energy  # (K - nc, h*w)

    # add repulsive energy
    repulsive_energy = compute_repulsive_energy(kpt_xyz_cur, chosen_mask, img_xyz, exp_factor=exp_factor)
    return (1 - repulsive_weight) * model_energy + repulsive_weight * repulsive_energy[non_chosen,:]


def compute_next_keypoint(
    kpt_xyz: torch.Tensor,
    kpts_chosen_mask: torch.Tensor,
    kpt_xyz_ref: torch.Tensor,
    candidates_xyz: torch.Tensor,
    kpt_likelihood_bias: torch.Tensor,
    constrain_mask: Optional[torch.Tensor] = None,
    use_model_energy=True,
    use_ambiguity_energy=True,
    repulsive_weight=0.5, 
    exp_factor=1000.0
):
    """
    Args:
        - kpt_xyz: (K, 3)
        - kpts_chosen_mask: (K,)
        - kpt_likelihood: (K, N)
        - kpt_xyz_ref: (K, 3)
        - candidates_xyz: (N, 3)
        - kpt_likelihood_bias: (K, N) Initial likelihood of keypoints at candidate locations
        - constrain_mask: (K,) mask to constrain the selection of keypoints
    """
    kpt_likelihood = kpt_likelihood_bias.clone()
    k, n = kpt_likelihood_bias.shape
    not_chosen = torch.logical_not(kpts_chosen_mask)
    kpt_indices = torch.arange(k)
    if constrain_mask is None:
        constrain_mask = torch.ones((k,), dtype=torch.bool)

    # 1) update kpt_likelihood of non-chosen keypoints with model energy
    model_energy, model_energy_likelihood = None, None
    if use_model_energy:
        model_energy = eval_energy_for_query_xyz(kpt_xyz, kpts_chosen_mask, candidates_xyz, kpt_xyz_ref, repulsive_weight=repulsive_weight, exp_factor=exp_factor) # (K - #chosen, N)
        model_energy_likelihood = torch.zeros_like(kpt_likelihood_bias)
        model_energy_likelihood[not_chosen,:] = torch.exp(-model_energy).float()
        model_energy_likelihood[torch.isnan(model_energy_likelihood)] = 0
        kpt_likelihood[not_chosen,:] *= model_energy_likelihood[not_chosen,:]

    # 2) update kpt_likelihood of non-chosen keypoints with ambiguity
    kpt_ambiguity_likelihood = None
    if use_ambiguity_energy:
        kpt_ambiguity_likelihood = torch.zeros_like(kpt_likelihood_bias)
        kpt_ambiguity_likelihood[not_chosen] = kpt_likelihood[not_chosen,:] / torch.sum(kpt_likelihood[not_chosen,:], dim=0, keepdim=True)
        kpt_ambiguity_likelihood[torch.isnan(kpt_ambiguity_likelihood)] = 0
        kpt_likelihood[not_chosen,:] *= kpt_ambiguity_likelihood[not_chosen,:]

    # 3) select non-chosen kpt_index at candidate location with maximum likelihood
    kpt_total_likelihood_argmax = torch.argmax(kpt_likelihood, dim=-1) # (K,)
    kpt_max_likelihoods = kpt_likelihood[kpt_indices, kpt_total_likelihood_argmax] # (K,)
    sort_res = torch.sort(kpt_max_likelihoods, descending=True)
    not_chosen_and_similar = torch.logical_and(not_chosen, constrain_mask)
    kpt_index = kpt_indices[sort_res.indices][not_chosen_and_similar[sort_res.indices]][0]
    kpt_candidate_index = kpt_total_likelihood_argmax[kpt_index]
    kpt_xyz = candidates_xyz[kpt_candidate_index,:]

    return int(kpt_index), int(kpt_candidate_index), kpt_xyz, kpt_likelihood, model_energy, model_energy_likelihood, kpt_ambiguity_likelihood, kpt_max_likelihoods, sort_res


def get_similar_keypoints(
    kpt_index: int, 
    kpt_candidate_index: int, 
    kpt_ambiguity_likelihood: torch.Tensor, 
    threshold: float = 0.4
):
    k, n = kpt_ambiguity_likelihood.shape
    kpt_indices = torch.arange(k)
    kpt_indices = kpt_indices[kpt_indices != kpt_index]
    sort_res = torch.sort(kpt_ambiguity_likelihood[kpt_indices, kpt_candidate_index], descending=True)
    kpt_indices = kpt_indices[sort_res.indices]
    kpt_ambiguities = sort_res.values
    mask = kpt_ambiguities > threshold
    return kpt_indices[mask], kpt_ambiguities[mask]


def compute_next_keypoint_with_ambiguity_check(
    kpt_likelihood_bias: torch.Tensor,
    img_xyz_orig: torch.Tensor,
    img_xyz: torch.Tensor,
    kpt_xyz_cur: torch.Tensor,
    kpt_xyz_ref: torch.Tensor,
    kpt_chosen: torch.Tensor,
    kpt_index: Optional[int] = None, 

    # params
    kpt_is_similar_threshold=0.4,
    peak_min_distance=10,
    max_num_peaks=5,
    min_peak_value=0.3
):
    constrain_mask = None
    if kpt_index is not None:
        constrain_mask = torch.zeros((kpt_likelihood_bias.shape[0],), dtype=torch.bool)
        constrain_mask[kpt_index] = True
    kpt_index, kpt_candidate_index, kpt_xyz, kpt_likelihood, _, _, kpt_ambiguity_likelihood, _, _ = compute_next_keypoint(
        kpt_xyz = kpt_xyz_cur,
        kpts_chosen_mask = kpt_chosen,
        kpt_xyz_ref = kpt_xyz_ref,
        candidates_xyz = img_xyz,
        kpt_likelihood_bias = kpt_likelihood_bias,
        constrain_mask = constrain_mask,
    )

    # Idea: check if chosen kpt_index still has similar keypoints, in which case we should test different locations for the selected kpt_index
    similar_kpt_indices, _ = get_similar_keypoints(
        kpt_index = kpt_index, 
        kpt_candidate_index = int(kpt_candidate_index), 
        kpt_ambiguity_likelihood= kpt_ambiguity_likelihood, 
        threshold=kpt_is_similar_threshold
    )
    if len(similar_kpt_indices) == 0:
        return kpt_index, kpt_xyz

    # extract peaks (can be made differentiable later on)
    embd_size = int(img_xyz.shape[0]**0.5)
    # TODO this will be simplified to simply take all xyz values that are above a certain threshold
    xyz_peak_coords, _, _ = ft_align_utils.extract_peaks_xyz(
        img_attn = kpt_likelihood[kpt_index].reshape((embd_size, embd_size)),
        img_xyz_orig=img_xyz_orig,
        max_num_peaks=max_num_peaks,
        min_distance=peak_min_distance,
        min_peak_value=min_peak_value
    )
    if xyz_peak_coords is None or xyz_peak_coords.shape[0] == 0:
        print("Warning: no peaks found")
        return kpt_index, kpt_xyz
    n_peaks = xyz_peak_coords.shape[0]

    # compute model energy
    n_similar = similar_kpt_indices.shape[0]
    model_energies = torch.zeros((n_peaks,), dtype=torch.float32)
    for p in range(n_peaks):
        kpt_xyz_cur_look = kpt_xyz_cur.clone()
        kpts_chosen_mask_cur_look = kpt_chosen.clone()

        # update kpt_xyz_cur_look and kpts_chosen_mask_cur_look with kpt_index
        kpt_xyz_cur_look[kpt_index,:] = xyz_peak_coords[p]
        kpts_chosen_mask_cur_look[kpt_index] = True
        
        for i in range(n_similar):
            kpt_index_i, kpt_xyz_i = compute_next_keypoint_with_ambiguity_check(
                kpt_index = int(similar_kpt_indices[i]),
                kpt_likelihood_bias = kpt_likelihood_bias,
                img_xyz_orig = img_xyz_orig,
                img_xyz = img_xyz,
                kpt_xyz_cur = kpt_xyz_cur_look,
                kpt_xyz_ref = kpt_xyz_ref,
                kpt_chosen = kpts_chosen_mask_cur_look,
            )
            # update kpt_xyz_cur_look and kpts_chosen_mask_cur_look with kpt_index
            kpt_xyz_cur_look[kpt_index_i,:] = kpt_xyz_i
            kpts_chosen_mask_cur_look[kpt_index_i] = True

        model_energies[p] = eval_energy_for_chosen(kpt_xyz_cur_look, kpts_chosen_mask_cur_look, kpt_xyz_ref)

    # select peak with minimum energy
    _, min_energy_idx = torch.min(model_energies, dim=0)
    kpt_xyz = xyz_peak_coords[min_energy_idx]

    return kpt_index, kpt_xyz



def my_algo(
    kpt_likelihood: torch.Tensor,
    kpt_xyz_ref: torch.Tensor,
    img_xyz: torch.Tensor
):

    """
    Args:
        - kpt_likelihood: (K, h, w)
        - kpt_xyz_ref: (K, 3)
        - img_xyz: (h*w, 3)
    Returns:
        - kpt_likelihood: (K, h, w)
        - kpt_xyz_cur: (K, 3)
    """
    kpt_likelihood_orig = kpt_likelihood.clone()
    kpt_likelihood_cur = kpt_likelihood.clone()
    K, h, w = kpt_likelihood_cur.shape
    chosen_mask = torch.zeros((K,), dtype=torch.bool)
    kpt_xyz_cur = torch.zeros_like(kpt_xyz_ref)
    kpt_indices = torch.arange(K)

    for i in tqdm(range(K)):
        # 1) compute total likelihood, which includes ambiguity
        kpt_ambiguity_likelihood = kpt_likelihood_cur / torch.sum(kpt_likelihood_cur, dim=0, keepdim=True)
        kpt_total_likelihood = kpt_likelihood_cur * kpt_ambiguity_likelihood

        # 2) select kpt from not-yet-chosen set with maximum likelihood
        kpt_max_likelihoods = torch.max(kpt_total_likelihood.reshape((kpt_likelihood_cur.shape[0],-1)), dim=-1)[0]
        not_chosen = torch.logical_not(chosen_mask)
        sort_res = torch.sort(kpt_max_likelihoods[not_chosen], descending=True)
        kpt_index = kpt_indices[not_chosen][sort_res.indices[0]]

        # 3) compute xyz for newly chosen kpt
        kpt_likelihood_flat = kpt_likelihood_cur.reshape((K,-1))
        kpt_likelihood_flat_sum = torch.sum(kpt_likelihood_flat, dim=-1)
        kpt_xyz_avg = torch.sum(kpt_likelihood_flat[:,:,None] * img_xyz[None,:,:], dim=1) / kpt_likelihood_flat_sum[:,None]
        kpt_xyz = kpt_xyz_avg[kpt_index,:]

        # 4) update kpt_xyz_cur and chosen_mask
        kpt_xyz_cur[kpt_index,:] = kpt_xyz
        chosen_mask[kpt_index] = True

        if i < K - 1:
            # 5) compute energy for each pixel and for each kpt
            # TODO add repulsive energy so that two kpt do not land at the same spot
            model_energy = eval_energy_for_query_xyz(kpt_xyz_cur, chosen_mask, img_xyz, kpt_xyz_avg)
            model_energy = model_energy.reshape((-1, h, w))

            # 6) update kpt_likelihood with model energy
            not_chosen = torch.logical_not(chosen_mask)
            kpt_likelihood_cur[not_chosen,:,:] = kpt_likelihood_orig[not_chosen,:,:] * torch.exp(-model_energy).float()

    return kpt_likelihood_cur, kpt_xyz_cur
