from scipy.interpolate import RegularGridInterpolator
import numpy as np
import torch
import utils.depth_utils as depth_utils
import utils.slam_utils as slam_utils
from lightglue import viz2d
from dataclasses import dataclass


def pixel_coords_to_colors(pixel_coords_xy: np.ndarray, image: np.ndarray):
    """
    Convert image coordinates to image colors.

    Parameters:
    - pixel_coords_xy: np.ndarray with shape (n, 2), representing the n image coordinates.
    - image: np.ndarray with shape (h, w, 3), representing the an RGB image.
    """
    x = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y = np.linspace(0, image.shape[1] - 1, image.shape[1])
    # TODO instead of interpolation find minimum across neighours
    # Or Alternatively take 
    interp_color_r = RegularGridInterpolator((x, y), image[:, :, 0], bounds_error=False, fill_value=0)
    interp_color_g = RegularGridInterpolator((x, y), image[:, :, 1], bounds_error=False, fill_value=0)
    interp_color_b = RegularGridInterpolator((x, y), image[:, :, 2], bounds_error=False, fill_value=0)

    color_r = interp_color_r(pixel_coords_xy @ np.array([[0, 1], [1, 0]]))[:, np.newaxis]
    color_g = interp_color_g(pixel_coords_xy @ np.array([[0, 1], [1, 0]]))[:, np.newaxis]
    color_b = interp_color_b(pixel_coords_xy @ np.array([[0, 1], [1, 0]]))[:, np.newaxis]


    pixel_colors = np.hstack((color_r, color_g, color_b))
    
    return pixel_colors


def is_on_geom(pixel_coords_xy, seg, ngeom):
    """ Check if pixel is on geom. Pixel can be on multiple geom (e.g. if pixel is on the boundary of two geom)
    Args:
        pixel_coords_xy: shape (n, 2)
        seg: shape (h, w)
        geom_xpos: shape (n_geom, 3)
        geom_xmat: shape (n_geom, 3, 3)
    Returns:
        is_on_geom: shape (n, n_geom) where is_on_geom[i, j] is True if pixel i is on geom j
    """

    x = np.linspace(0, seg.shape[0] - 1, seg.shape[0])
    y = np.linspace(0, seg.shape[1] - 1, seg.shape[1])
    interp_seg = RegularGridInterpolator((x, y), seg, bounds_error=False, fill_value=-1)

    pixel_coords_xy_center = np.floor(pixel_coords_xy)
    offsets = np.array([[0,0], [0,1], [1,0], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]])
    pixel_coords_xy_region = pixel_coords_xy_center[:, np.newaxis, :] + offsets[np.newaxis, :, :] # shape (n, 9, 2)

    geom_id_pixel_region = np.zeros((pixel_coords_xy.shape[0], 9), dtype=np.int64)
    for i in range(9):
        geom_id_pixel_region[:,i] = interp_seg(pixel_coords_xy_region[:, i, :] @ np.array([[0, 1], [1, 0]]))  # shape (n,)
    geom_id_pixel_region = geom_id_pixel_region.astype(np.int64)

    is_on_geom = np.zeros((pixel_coords_xy.shape[0], ngeom), dtype=bool)
    for i in range(ngeom):
        is_on_geom[:, i] = np.any(geom_id_pixel_region == i, axis=1)

    return is_on_geom


def rel_pos_to_geom(world_coords: np.ndarray, geom_xpos: np.ndarray, geom_xmat: np.ndarray):
    """ Compute relative position of points to geoms
    Parameters:
        world_coords: (N, 3) array of world coordinates
        geom_xpos: (M, 3) array of geom centers
        geom_xmat: (M, 3, 3) array of geom rot matrix
    Returns:
        rel_pos_to_blocks: (N, M, 3) array of relative positions
    """
    centered_world_coords = world_coords[:, None, :] - geom_xpos[None, :, :]
    rel_pos_to_blocks = np.matmul(geom_xmat.transpose((0, 2, 1)), centered_world_coords[:,:,:,None])
    # rel_pos_to_blocks = np.einsum("ijk,ikl->ijl", centered_world_coords, geom_xmat)
    return np.squeeze(rel_pos_to_blocks, axis=-1)


def match_world_coords(world_coords_0, world_coords_1, geom_xpos_0, geom_xpos_1, geom_xmat_0, geom_xmat_1, is_on_geom_0, is_on_geom_1, eps=0.01):
    """ Match world coordinates of keypoints in two images
    Parameters:
        world_coords_0: (N1, 3) array of world coordinates
        world_coords_1: (N2, 3) array of world coordinates
        geom_xpos_0: (M, 3) array of geom centers
        geom_xpos_1: (M, 3) array of geom centers
        geom_xmat_0: (M, 9) array of geom rot matrix
        geom_xmat_1: (M, 9) array of geom rot matrix
        is_on_geom_0: (N1, M) array of booleans
        is_on_geom_1: (N2, M) array of booleans
    Returns:
        matches_0: (N1,) array of indices of matches in world_coords_1
        matches_1: (N2,) array of indices of matches in world_coords_0
        assignment_mtr: (N1, N2) array of booleans indicating if a match is found
    """
    rel_pos_to_blocks_1 = rel_pos_to_geom(world_coords_0, geom_xpos_0, geom_xmat_0.reshape(-1, 3, 3))
    rel_pos_to_blocks_2 = rel_pos_to_geom(world_coords_1, geom_xpos_1, geom_xmat_1.reshape(-1, 3, 3))

    n_geoms = geom_xpos_0.shape[0]

    n1, n2 = is_on_geom_0.shape[0], is_on_geom_1.shape[0]
    global_index_0 = np.arange(n1)
    global_index_1 = np.arange(n2)

    assignment_mtr = np.zeros((n1, n2), dtype=bool)
    # in glue -1 means no match
    matches_0 = -np.ones(n1, dtype=np.int64)
    matches_1 = -np.ones(n2, dtype=np.int64)

    for i in range(n_geoms):
        is_on_geom_i_0 = is_on_geom_0[:, i]
        is_on_geom_i_1 = is_on_geom_1[:, i]
        if(np.sum(is_on_geom_i_0) == 0 or np.sum(is_on_geom_i_1) == 0):
            continue
        if np.linalg.norm(geom_xpos_0[i] - geom_xpos_1[i]) < 0.001:
            continue
        rel_pos_i_0 = rel_pos_to_blocks_1[is_on_geom_i_0, i, :]
        rel_pos_i_1 = rel_pos_to_blocks_2[is_on_geom_i_1, i, :]
        index_map_0 = global_index_0[is_on_geom_i_0]
        index_map_1 = global_index_1[is_on_geom_i_1]
        # compute distances between rel positions
        distances = np.linalg.norm(rel_pos_i_0[:, None, :] - rel_pos_i_1[None, :, :], axis=2)  # (n1, n2)
        # find best match
        min_dist = np.min(distances, axis=1) # (n1)
        argmin_dist = np.argmin(distances, axis=1)
        has_match = min_dist < eps
        if np.sum(has_match) == 0:
            continue
        # TODO remove all points for which points in A are assigned to the same point in B
        # TODO makes sure minimum holds in both directions
        match_index_0 = index_map_0[has_match]
        match_index_1 = index_map_1[argmin_dist[has_match]]
        matches_0[match_index_0] = match_index_1
        matches_1[match_index_1] = match_index_0
        assignment_mtr[match_index_0, match_index_1] = True

    # NOTE to make it compatible with glue
    # matches_1 = np.stack([global_index_1, matches_1], axis=1)
    # matches_2 = np.stack([global_index_2, matches_2], axis=1)
    # matches_1 = matches_1[matches_1[:, 1] != -1]
    # matches_2 = matches_2[matches_2[:, 1] != -1]

    return matches_0, matches_1, assignment_mtr


def create_sample_mask(is_on_geom, max_keypoints):
    num_kpts_per_geom = np.sum(is_on_geom, axis=0)
    kpt_weights = 1 + is_on_geom * (1 / (np.maximum(num_kpts_per_geom, 1)))[None, :]
    kpt_weights = np.max(kpt_weights, axis=-1)
    kpt_weights = kpt_weights / np.sum(kpt_weights)
    selected_indices = np.random.choice(
        np.arange(kpt_weights.shape[0]), 
        size=max_keypoints, 
        replace=False, 
        p=kpt_weights
    )
    mask = np.zeros(is_on_geom.shape[0], dtype=bool)
    mask[selected_indices] = True
    return mask


@dataclass
class MatchData:
    keypoints_0: torch.Tensor
    keypoints_1: torch.Tensor
    descriptors_0: torch.Tensor
    descriptors_1: torch.Tensor
    matches_0: torch.Tensor
    matches_1: torch.Tensor
    assignment_mtr: torch.Tensor


def match_features(
    T_pixel2world, 
    img_0, img_1,
    depth_0, depth_1, 
    seg_0, seg_1, 
    geom_xpos_0, geom_xpos_1, 
    geom_xmat_0, geom_xmat_1,
    kpts_0, kpts_1,
    dscpt_0, dscpt_1,
    ngeom,
    max_keypoints=200,
    eps=0.01 # distance threshold for matching
):
    # world coordinates
    world_coords_0, _, mask_moving_points_0, kpts_0_adjusted = depth_utils.pixel_coords_to_world_coords_moving_pixels(T_pixel2world, depth_0, depth_1, kpts_0)
    world_coords_1, _ , mask_moving_points_1, kpts_1_adjusted = depth_utils.pixel_coords_to_world_coords_moving_pixels(T_pixel2world, depth_1, depth_0, kpts_1)
    # world_coords_0_in_1, _ = depth_utils.pixel_coords_to_world_coords(T_pixel2world, depth_1, kpts_0)
    # world_coords_1_in_0, _ = depth_utils.pixel_coords_to_world_coords(T_pixel2world, depth_0, kpts_1)
    world_coords_0 = world_coords_0[:, :3]
    world_coords_1 = world_coords_1[:, :3]
    # world_coords_0_in_1 = world_coords_0_in_1[:, :3]
    # world_coords_1_in_0 = world_coords_1_in_0[:, :3]

    # colors
    # colors_0 = pixel_coords_to_colors(kpts_0, img_0)
    # colors_1 = pixel_coords_to_colors(kpts_1, img_1)
    # colors_0_in_1 = pixel_coords_to_colors(kpts_0, img_1)
    # colors_1_in_0 = pixel_coords_to_colors(kpts_1, img_0)

    # filter keypoints based on depth and color change + because of bug ignore keypoints of robot arm
    # NOTE because of this filter image 1 index and image 2 index may not be the same
    # mask_moving_points_0 = np.linalg.norm(world_coords_0_in_1 - world_coords_0, axis=1) > 0.01
    # mask_moving_points_1 = np.linalg.norm(world_coords_1_in_0 - world_coords_1, axis=1) > 0.01
    # TODO why do we need color change again?
    # mask_color_change_0 = np.linalg.norm(colors_0_in_1 - colors_0, axis=1) > 0.01
    # mask_color_change_1 = np.linalg.norm(colors_1_in_0 - colors_1, axis=1) > 0.01 
    mask_0 = mask_moving_points_0 # np.logical_or(mask_moving_points_0, mask_color_change_0)
    mask_1 = mask_moving_points_1 # np.logical_or(mask_moving_points_1, mask_color_change_1)

    # apply filter
    # TODO make sure to keep keypoints from all geoms => or apply max keypoints after matching keypoints
    world_coords_0 = world_coords_0[mask_0]
    world_coords_1 = world_coords_1[mask_1]
    kpts_0 = kpts_0[mask_0]
    kpts_1 = kpts_1[mask_1]
    kpts_0_adjusted = kpts_0_adjusted[mask_0]
    kpts_1_adjusted = kpts_1_adjusted[mask_1]
    dscpt_0 = dscpt_0[mask_0]
    dscpt_1 = dscpt_1[mask_1]

    # TODO we can combine these two steps
    is_on_geom_0 = is_on_geom(kpts_0_adjusted, seg_0, ngeom)
    is_on_geom_1 = is_on_geom(kpts_1_adjusted, seg_1, ngeom)

    sample_mask_0 = create_sample_mask(is_on_geom_0, np.minimum(max_keypoints, kpts_0.shape[0]))
    sample_mask_1 = create_sample_mask(is_on_geom_1, np.minimum(max_keypoints, kpts_1.shape[0]))
    is_on_geom_0 = is_on_geom_0[sample_mask_0]
    is_on_geom_1 = is_on_geom_1[sample_mask_1]
    world_coords_0 = world_coords_0[sample_mask_0]
    world_coords_1 = world_coords_1[sample_mask_1]
    kpts_0 = kpts_0[sample_mask_0]
    kpts_1 = kpts_1[sample_mask_1]
    dscpt_0 = dscpt_0[sample_mask_0]
    dscpt_1 = dscpt_1[sample_mask_1]
    
    matches_0, matches_1, assignment_mtr = match_world_coords(
        world_coords_0, world_coords_1, 
        geom_xpos_0, 
        geom_xpos_1, 
        geom_xmat_0, 
        geom_xmat_1, 
        is_on_geom_0, 
        is_on_geom_1,
        eps=eps
    )

    # convert to tensors
    kpts_0 = torch.from_numpy(kpts_0).float()
    kpts_1 = torch.from_numpy(kpts_1).float()
    dscpt_0 = torch.from_numpy(dscpt_0).float()
    dscpt_1 = torch.from_numpy(dscpt_1).float()
    matches_0 = torch.from_numpy(matches_0).long()
    matches_1 = torch.from_numpy(matches_1).long()
    assignment_mtr = torch.from_numpy(assignment_mtr).bool()

    # padding
    # pad keypoints with 0
    kpts_0 = torch.cat([kpts_0, torch.zeros((max_keypoints - kpts_0.shape[0], 2))], dim=0)
    kpts_1 = torch.cat([kpts_1, torch.zeros((max_keypoints - kpts_1.shape[0], 2))], dim=0)
    # pad descriptors with 0
    dscpt_0 = torch.cat([dscpt_0, torch.zeros((max_keypoints - dscpt_0.shape[0], 256))], dim=0)
    dscpt_1 = torch.cat([dscpt_1, torch.zeros((max_keypoints - dscpt_1.shape[0], 256))], dim=0)
    # pad matches with -1
    matches_0 = torch.cat([matches_0, -torch.ones(max_keypoints - matches_0.shape[0], dtype=torch.long)])
    matches_1 = torch.cat([matches_1, -torch.ones(max_keypoints - matches_1.shape[0], dtype=torch.long)])
    # pad assignment_mtr with False
    assignment_mtr = torch.cat([assignment_mtr, torch.zeros((max_keypoints - assignment_mtr.shape[0], assignment_mtr.shape[1]), dtype=torch.bool)], dim=0)
    assignment_mtr = torch.cat([assignment_mtr, torch.zeros((max_keypoints, max_keypoints - assignment_mtr.shape[1]), dtype=torch.bool)], dim=1)

    args = {
        "keypoints_0": kpts_0,
        "keypoints_1": kpts_1,
        "descriptors_0": dscpt_0,
        "descriptors_1": dscpt_1,
        "matches_0": matches_0,
        "matches_1": matches_1,
        "assignment_mtr": assignment_mtr,
    }
    return args


def match_keyframes(keyframe_data_0, keyframe_data_1, extractor, T_pixel2world, ngeom, k0=0, k1=0, eps=0.01, max_keypoints=200):
    img_0 = keyframe_data_0["img"][k0]
    img_1 = keyframe_data_1["img"][k1]
    seg_0 = keyframe_data_0["seg"][k0,:,:,0] # seg consist of geom id and body type
    seg_1 = keyframe_data_1["seg"][k1,:,:,0]
    depth_0 = keyframe_data_0["depth"][k0]
    depth_1 = keyframe_data_1["depth"][k1]
    fts_0 = slam_utils.compute_features(extractor, img_0)
    fts_1 = slam_utils.compute_features(extractor, img_1)
    kpts_0 = fts_0["keypoints"].cpu().numpy()
    kpts_1 = fts_1["keypoints"].cpu().numpy()

    # match features
    geom_xpos_0 = keyframe_data_0["geom_xpos"][k0]
    geom_xpos_1 = keyframe_data_1["geom_xpos"][k1]
    geom_xmat_0 = keyframe_data_0["geom_xmat"][k0]
    geom_xmat_1 = keyframe_data_1["geom_xmat"][k1]
    dscpt_0 = fts_0["descriptors"].cpu().numpy()
    dscpt_1 = fts_1["descriptors"].cpu().numpy()
    match_data = match_features(
        T_pixel2world,
        img_0, img_1,
        depth_0, depth_1,
        seg_0, seg_1,
        geom_xpos_0, geom_xpos_1,
        geom_xmat_0, geom_xmat_1,
        kpts_0, kpts_1,
        dscpt_0, dscpt_1,
        ngeom=ngeom,
        max_keypoints=max_keypoints,
        eps=eps
    )
    return match_data


def visualize_matches(keyframe_data_0, keyframe_data_1, match_data, k0=0, k1=0, title=""):
    img_0 = keyframe_data_0["img"][k0]
    img_1 = keyframe_data_1["img"][k1]
    axes = viz2d.plot_images([img_0, img_1], titles=[title + " img_0", title + " img_1"])
    m_0 = match_data["matches_0"].cpu().numpy()
    k_0 = match_data["keypoints_0"].cpu().numpy()
    k_1 = match_data["keypoints_1"].cpu().numpy()
    m_kpts_0 = k_0[m_0 != -1]
    m_kpts_1 = k_1[m_0[m_0 != -1]]
    viz2d.plot_matches(m_kpts_0, m_kpts_1, color="lime", lw=0.2)


@dataclass
class ImageData:
    img: torch.Tensor
    depth: torch.Tensor
    seg: torch.Tensor
    geom_xpos: torch.Tensor
    geom_xmat: torch.Tensor
    keypoints: torch.Tensor
    keypoint_scores: torch.Tensor
    descriptors: torch.Tensor

def keyframe_data_to_image_data(keyframe_data, k):
    args = {
        "img": keyframe_data["img"][k],
        "depth": keyframe_data["depth"][k],
        "seg": keyframe_data["seg"][k],
        "geom_xpos": keyframe_data["geom_xpos"][k],
        "geom_xmat": keyframe_data["geom_xmat"][k],
        "keypoints": keyframe_data["keypoints"][k],
        "keypoint_scores": keyframe_data["keypoint_scores"][k],
        "descriptors": keyframe_data["descriptors"][k],
    }
    return ImageData(**args)