from turtle import st
import torch
import numpy as np
from dataclasses import dataclass
import yaml

from lightglue import viz2d

import utils.match_utils as match_utils
import utils.depth_utils as depth_utils
import utils.predict_utils as predict_utils
import utils.torch_utils as torch_utils
import utils.slam_utils as slam_utils

DEVICE = "cuda"


def load_config(
    sub_project_name, config_id, ROOT_DIR="/home/user/Documents/projects/Metaworld"
):
    SUB_PROJECT_NAME = sub_project_name
    CONFIG_IDX = config_id

    # do not change
    SUB_PROJECT_DIR = f"{ROOT_DIR}/keyframes/{SUB_PROJECT_NAME}"
    CONFIG_PATH = (
        f"{SUB_PROJECT_DIR}/configs/config_{SUB_PROJECT_NAME}_{CONFIG_IDX}.yaml"
    )

    # load config

    with open(CONFIG_PATH, mode="rt", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # validate config
    assert (
        config["dataset_args"] is not None
        and config["model_args"] is not None
        and config["optimizer_args"] is not None
        and config["training"] is not None
    )

    return config


@dataclass
class ImageData:
    image: torch.Tensor
    depth: torch.Tensor
    block_centers: torch.Tensor
    block_shapes: torch.Tensor
    keypoints: torch.Tensor
    keypoint_scores: torch.Tensor
    descriptors: torch.Tensor


def keyframe_data_to_image_data(keyframe_data, k):
    args = {
        "image": keyframe_data["keyframe_images"][k],
        "depth": keyframe_data["keyframe_depth_images"][k],
        "block_centers": keyframe_data["keyframe_block_centers"][k],
        "block_shapes": keyframe_data["keyframe_block_shapes"],
        "keypoints": keyframe_data["keyframe_keypoints"][k],
        "keypoint_scores": keyframe_data["keyframe_keypoint_scores"][k],
        "descriptors": keyframe_data["keyframe_descriptors"][k],
    }
    return ImageData(**args)


@dataclass
class MatchData:
    keypoints_0: torch.Tensor
    keypoints_1: torch.Tensor
    descriptors_0: torch.Tensor
    descriptors_1: torch.Tensor
    matches_0: torch.Tensor
    matches_1: torch.Tensor
    assignment_mtr: torch.Tensor


def show_matches(image_1: torch.Tensor, image_2: torch.Tensor, match_data: MatchData):
    image_1_np = image_1.numpy() if type(image_1) == torch.Tensor else image_1
    image_2_np = image_2.numpy() if type(image_2) == torch.Tensor else image_2
    keypoints_1_np = (
        match_data.keypoints_0.numpy()
        if type(match_data.keypoints_0) == torch.Tensor
        else match_data.keypoints_0
    )
    keypoints_2_np = (
        match_data.keypoints_1.numpy()
        if type(match_data.keypoints_1) == torch.Tensor
        else match_data.keypoints_1
    )

    matches = []
    matches_1_np = (
        match_data.matches_0.numpy()
        if type(match_data.matches_0) == torch.Tensor
        else match_data.matches_0
    )
    for i in range(matches_1_np.shape[0]):
        if matches_1_np[i] != -1:
            matches.append([i, matches_1_np[i]])
    matches = np.array(matches)

    m_kpts0 = keypoints_1_np[matches[:, 0]]
    m_kpts1 = keypoints_2_np[matches[:, 1]]
    axes = viz2d.plot_images([image_1_np, image_2_np])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)


def show_matches_kpts(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    keypoints_1_np: np.ndarray,
    keypoints_2_np: np.ndarray,
):
    image_1_np = image_1.numpy() if type(image_1) == torch.Tensor else image_1
    image_2_np = image_2.numpy() if type(image_2) == torch.Tensor else image_2
    axes = viz2d.plot_images([image_1_np, image_2_np])
    viz2d.plot_matches(keypoints_1_np, keypoints_2_np, color="lime", lw=0.2)


def show_keypoints(image_data: ImageData, keypoints_np=None):
    image_np = image_data.image.numpy()
    if keypoints_np is None:
        keypoints_np = image_data.keypoints.numpy()
    axes = viz2d.plot_images([image_np])
    viz2d.plot_keypoints([keypoints_np])


def predictions_to_match_data(match_data_gt: MatchData, predictions, b=0):
    # see pl_module pre_step
    args_pred = {
        "keypoints_0": match_data_gt.keypoints_0,
        "keypoints_1": match_data_gt.keypoints_1,
        "descriptors_0": match_data_gt.descriptors_0,
        "descriptors_1": match_data_gt.descriptors_1,
        "matches_0": predictions["matches0"][b],
        "matches_1": predictions["matches1"][b],
        "assignment_mtr": match_data_gt.assignment_mtr,
    }
    match_data_pred = MatchData(**args_pred)
    return match_data_pred


@dataclass
class MatchDataNP:
    keypoints_0: np.ndarray
    keypoints_1: np.ndarray
    descriptors_0: np.ndarray
    descriptors_1: np.ndarray
    matches_0: np.ndarray
    matches_1: np.ndarray
    assignment_mtr: np.ndarray


def predictions_to_match_data_v2(model_input, predictions, b=0):
    # see pl_module pre_step
    args_pred = {
        "keypoints_0": model_input["keypoints0"][b],
        "keypoints_1": model_input["keypoints1"][b],
        "descriptors_0": model_input["descriptors0"][b],
        "descriptors_1": model_input["descriptors1"][b],
        "matches_0": predictions["matches0"][b],
        "matches_1": predictions["matches1"][b],
        # TODO use assignment_mtr from predictions!!
        "assignment_mtr": model_input["gt_assignment"][b],
    }
    args_pred = torch_utils.to_numpy(args_pred)
    match_data_pred = MatchDataNP(**args_pred)
    return match_data_pred


def batch_data_to_match_data(batch, b=0):
    if len(batch) == 14:
        (
            img_0,
            img_1,
            depth_0,
            depth_1,
            seg_0,
            seg_1,
            match_data,
            keypoints_1,
            keypoints_2,
            descriptors_1,
            descriptors_2,
            matches_1,
            matches_2,
            assignment_mtr,
        ) = batch
    else:
        (
            keypoints_1,
            keypoints_2,
            descriptors_1,
            descriptors_2,
            matches_1,
            matches_2,
            assignment_mtr,
        ) = batch
    args = {
        "keypoints_0": keypoints_1[b],
        "keypoints_1": keypoints_2[b],
        "descriptors_0": descriptors_1[b],
        "descriptors_1": descriptors_2[b],
        "matches_0": matches_1[b],
        "matches_1": matches_2[b],
        "assignment_mtr": assignment_mtr[b],
    }
    return MatchData(**args)


def match_data_to_model_input(match_data: MatchData, image_size=(360, 480)):
    args = {
        "keypoints0": match_data.keypoints_0[None, :],
        "descriptors0": match_data.descriptors_0[None, :],
        "keypoints1": match_data.keypoints_1[None, :],
        "descriptors1": match_data.descriptors_1[None, :],
        "view0": {"image_size": image_size},
        "view1": {"image_size": image_size},
        "gt_matches0": match_data.matches_0[None, :],
        "gt_matches1": match_data.matches_1[None, :],
        "gt_assignment": match_data.assignment_mtr[None, :],
    }
    return args


def get_model_input_v2(
    kpts_0: torch.Tensor,
    kpts_1: torch.Tensor,
    desc_0: torch.Tensor,
    desc_1: torch.Tensor,
    image_size=(360, 480),
    device="cuda",
):
    matches_0 = -torch.ones(kpts_0.shape[0])
    matches_1 = -torch.ones(kpts_1.shape[0])
    assignment_mtr = torch.zeros((kpts_0.shape[0], kpts_1.shape[0]))
    args = {
        "keypoints0": kpts_0[None, :].to(device),
        "descriptors0": desc_0[None, :].to(device),
        "keypoints1": kpts_1[None, :].to(device),
        "descriptors1": desc_1[None, :].to(device),
        "view0": {"image_size": image_size},
        "view1": {"image_size": image_size},
        # dummy data
        "gt_matches0": matches_0[None, :].to(device),
        "gt_matches1": matches_1[None, :].to(device),
        "gt_assignment": assignment_mtr[None, :].to(device),
    }
    return args


def get_model_input_v3_np(
    kpts_0: np.ndarray,
    kpts_1: np.ndarray,
    desc_0: np.ndarray,
    desc_1: np.ndarray,
    image_size=(360, 480),
    device="cuda",
):
    matches_0 = -torch.ones(kpts_0.shape[0])
    matches_1 = -torch.ones(kpts_1.shape[0])
    assignment_mtr = torch.zeros((kpts_0.shape[0], kpts_1.shape[0]))
    args = {
        "keypoints0": torch.from_numpy(kpts_0[None, :]).to(device),
        "descriptors0": torch.from_numpy(desc_0[None, :]).to(device),
        "keypoints1": torch.from_numpy(kpts_1[None, :]).to(device),
        "descriptors1": torch.from_numpy(desc_1[None, :]).to(device),
        "view0": {"image_size": image_size},
        "view1": {"image_size": image_size},
        # dummy data
        "gt_matches0": matches_0[None, :].to(device),
        "gt_matches1": matches_1[None, :].to(device),
        "gt_assignment": assignment_mtr[None, :].to(device),
    }
    return args


# ====================
# Predict next keypoint position
# ====================


def get_close_feature_points(
    query_world_coord: torch.Tensor,
    world_coords: torch.Tensor,
    ignore: torch.Tensor,
    n_closest=1,
):
    """Get closest feature points to query_world_coord
    Args:
        query_world_coord (torch.Tensor): (3,) tensor of query world coordinate
        world_coords (torch.Tensor): (N, 3) tensor of world coordinates
        ignore (torch.Tensor): (N,) tensor of bools to ignore some points
        n_closest (int, optional): number of closest points to return. Defaults to 1.
    Returns:
        closes_point_indices (torch.Tensor): (n_closest,) tensor of indices of closest points
    """
    # compute distance
    dist = torch.norm(world_coords - query_world_coord, dim=1)
    # ignore
    dist[ignore] = torch.tensor(float("inf"))
    # get closest points
    _, closest_point_indices = torch.topk(dist, k=n_closest, largest=False)
    return closest_point_indices


# def get_next_keypoint_positions(
#     image_data_1_1: ImageData,
#     image_data_1_2: ImageData,
#     image_data_2_1: ImageData,
#     match_data_1_1_n_1_2: MatchData,
#     match_data_1_1_n_2_1: MatchData,
#     point_cloud_sensor: pc_sensor.PointCloudSensor,
# ):
#     # get world coordinates
#     kpts_1_in_1_1 = env_utils.pixel_coords_to_world_coords(
#         point_cloud_sensor,
#         match_data_1_1_n_1_2.keypoints_1.numpy(),
#         image_data_1_1.depth.numpy()
#     )[:,:3]
#     kpts_2_in_1_2 = env_utils.pixel_coords_to_world_coords(
#         point_cloud_sensor,
#         match_data_1_1_n_1_2.keypoints_2.numpy(),
#         image_data_1_2.depth.numpy()
#     )[:,:3]


#     world_coords_1_2 = env_utils.pixel_coords_to_world_coords(
#         point_cloud_sensor,
#         image_data_1_2.keypoints.numpy(),
#         image_data_1_2.depth.numpy()
#     )[:,:3]
#     world_coords_2_1 = env_utils.pixel_coords_to_world_coords(
#         point_cloud_sensor,
#         image_data_2_1.keypoints.numpy(),
#         image_data_2_1.depth.numpy()
#     )[:,:3]

#     # get keypoints indices that are moving between 1_1 and 1_2

#     pass


# cluster keypoints based on points that have same distance in both frames
def cluster_keypoints(
    world_coords_1: torch.Tensor, world_coords_2: torch.Tensor, match_data: MatchData
):
    # world coordinates

    pass


# get closest cluster to moving brick in keyframes
def get_closest_cluster():
    pass


# search moving and cloest cluster in current environment

# move block accordingly


def match_lightglue_finetuned(model, match_data_gt: match_utils.MatchData):
    pred = model(match_data_to_model_input(match_data_gt))
    match_data_pred = predictions_to_match_data(match_data_gt, pred, 0)
    return match_data_pred


def match_lightglue_finetuned_v2(model, model_input):
    pred = model(model_input)
    match_data_pred = predictions_to_match_data_v2(model_input, pred, 0)
    return match_data_pred


def predict_feature_movement(
    model,
    extractor,
    ngeom,
    keyframe_data_0,
    keyframe_data_1,
    T_world2pixel,
    T_pixel2world,
    k=0,
):
    # image data
    image_data_0_0 = match_utils.keyframe_data_to_image_data(keyframe_data_0, k)
    image_data_0_1 = match_utils.keyframe_data_to_image_data(keyframe_data_0, k + 1)
    image_data_1_0 = image_data_0_0
    image_data_1_1 = match_utils.keyframe_data_to_image_data(keyframe_data_1, k)

    # match data
    match_data_0_gt_args = match_utils.match_keyframes(
        keyframe_data_0,
        keyframe_data_0,
        extractor,
        T_pixel2world,
        ngeom,
        k0=0,
        k1=1,
        eps=0.01,
        max_keypoints=300,
    )
    match_data_1_gt_args = match_utils.match_keyframes(
        keyframe_data_0,
        keyframe_data_1,
        extractor,
        T_pixel2world,
        ngeom,
        k0=0,
        k1=0,
        eps=0.01,
        max_keypoints=300,
    )
    match_data_0_gt = match_utils.MatchData(**match_data_0_gt_args)
    match_data_1_gt = match_utils.MatchData(**match_data_1_gt_args)
    # match_data_1_gt = inference_utils.match_images(image_data_1_1, image_data_1_2, keyframe_data_1["point_cloud_sensor"])
    # match_data_2_gt = inference_utils.match_images(image_data_2_1, image_data_2_2, keyframe_data_1["point_cloud_sensor"])
    match_data_0 = match_lightglue_finetuned(model, match_data_0_gt)
    # match_data_1 = match_data_1_gt
    match_data_1 = match_lightglue_finetuned(model, match_data_1_gt)

    # world coords for keypoints
    wpos_kpts_0_0 = depth_utils.pixel_coords_to_world_coords(
        T_pixel2world, image_data_0_0.depth, match_data_0.keypoints_0.numpy()
    )[:, :3]
    wpos_kpts_0_1 = depth_utils.pixel_coords_to_world_coords(
        T_pixel2world, image_data_0_1.depth, match_data_0.keypoints_1.numpy()
    )[:, :3]
    wpos_kpts_1_0 = depth_utils.pixel_coords_to_world_coords(
        T_pixel2world, image_data_1_0.depth, match_data_1.keypoints_0.numpy()
    )[:, :3]
    wpos_kpts_1_1 = depth_utils.pixel_coords_to_world_coords(
        T_pixel2world, image_data_1_1.depth, match_data_1.keypoints_1.numpy()
    )[:, :3]

    # inference_utils.show_matches(image_data_0_0.img, image_data_0_1.img, match_data_0)
    # inference_utils.show_matches(image_data_1_0.img, image_data_1_1.img, match_data_1)

    # map kpts_0_0 to kpts_1_0
    m_0_0_to_1_0 = -np.ones_like(match_data_0.matches_0)
    for idx_0_0 in range(match_data_0.matches_0.shape[0]):
        wpos_0_0 = wpos_kpts_0_0[idx_0_0]
        dists = np.linalg.norm(wpos_0_0 - wpos_kpts_1_0, axis=1)
        idx_1_0 = np.argmin(dists)
        if dists[idx_1_0] < 0.05:
            m_0_0_to_1_0[idx_0_0] = idx_1_0

    # find moving keypoints in match_data_0
    m_moving_0_0_to_0_1 = []
    for idx_0_0 in range(match_data_0.matches_0.shape[0]):
        idx_0_1 = match_data_0.matches_0[idx_0_0]
        if idx_0_1 != -1:
            wpos_0 = wpos_kpts_0_0[idx_0_0]
            wpos_1 = wpos_kpts_0_1[idx_0_1]
            if np.linalg.norm(wpos_0 - wpos_1) > 0.025:
                m_moving_0_0_to_0_1.append((idx_0_0, idx_0_1))
    m_moving_0_0_to_0_1 = np.array(m_moving_0_0_to_0_1)

    # we assume only 1 object is moving between frames
    # compute center of moving points in 0_1
    wcenter_moving_0_1 = np.mean(wpos_kpts_0_1[m_moving_0_0_to_0_1[:, 1]], axis=0)

    # find points in 0_0 close to wcenter_moving_0_1
    closest_points_in_1_0 = predict_utils.get_close_feature_points_np(
        wcenter_moving_0_1, wpos_kpts_1_0, ignore=None, n_closest=10
    )

    # create map for moving keypoints in 0_1 to 1_1
    m_moving_0_1_to_1_1 = []
    for i in range(m_moving_0_0_to_0_1.shape[0]):
        idx_0_0 = m_moving_0_0_to_0_1[i][0]
        idx_0_1 = m_moving_0_0_to_0_1[i][1]
        idx_1_0 = m_0_0_to_1_0[idx_0_0]
        if idx_1_0 != -1:
            idx_1_1 = match_data_1.matches_0[idx_1_0]
            if idx_1_1 != -1:
                m_moving_0_1_to_1_1.append((idx_0_1, idx_1_1))
    m_moving_0_1_to_1_1 = np.array(m_moving_0_1_to_1_1)

    # TODO fix description
    # create map for closest keypoints in 1_1 to 2_1
    m_closest_1_0_to_1_1 = []
    for i in range(closest_points_in_1_0.shape[0]):
        idx_1_0 = closest_points_in_1_0[i]
        idx_1_1 = match_data_1.matches_0[idx_1_0]
        if idx_1_1 != -1:
            m_closest_1_0_to_1_1.append((idx_1_0, idx_1_1))
    m_closest_1_0_to_1_1 = np.array(m_closest_1_0_to_1_1)

    # compute transformation of closest point matches
    # R_closest, t_closest, t_no_rot_closest, x_center, predict_closest = predict_utils.compute_T(
    #     wpos_kpts_1_0[m_closest_1_0_to_1_1[:,0]],
    #     wpos_kpts_1_1[m_closest_1_0_to_1_1[:,1]]
    # )
    t_closest, predict_closest = predict_utils.estimate_translation(
        wpos_kpts_1_0[m_closest_1_0_to_1_1[:, 0]],
        wpos_kpts_1_1[m_closest_1_0_to_1_1[:, 1]],
    )

    # compute new positions of moving points in 2_1
    kpts_2_0 = match_data_1.keypoints_1[m_moving_0_1_to_1_1[:, 1]]
    wpos_kpts_2_0 = wpos_kpts_1_1[m_moving_0_1_to_1_1[:, 1]]
    wpos_kpts_2_1 = predict_closest(wpos_kpts_0_1[m_moving_0_1_to_1_1[:, 0]])
    kpts_2_1 = depth_utils.world_coords_to_pixel_coords(T_world2pixel, wpos_kpts_2_1)
    # kpts_3_2 = env_utils.world_coords_to_pixel_coords(point_cloud_sensor, wpos_kpts_2_2[m_moving_1_2_to_2_2[:,1]])

    # TODO
    show_matches_kpts(
        image_data_1_1.img, image_data_1_1.img, kpts_2_0.numpy(), kpts_2_1
    )

    return {
        "T": {
            # "R_closest": R_closest,
            "t_closest": t_closest,
            # "t_no_rot_closest": t_no_rot_closest,
            # "x_center": x_center,
        },
        "kpts": {
            "kpts_2_0": kpts_2_0,
            "kpts_2_1": kpts_2_1,
            "wpos_kpts_2_0": wpos_kpts_2_0,
            "wpos_kpts_2_1": wpos_kpts_2_1,
        },
    }


# TODO for simplicity, maybe it would be better to use numpy only
def predict_feature_movement_v2(
    model_ft,
    depth_ref_0: np.ndarray,
    depth_ref_1: np.ndarray,
    depth_cur: np.ndarray,
    desc_ref_0: np.ndarray,
    desc_cur: np.ndarray,
    kpts_ref_0: np.ndarray,
    kpts_ref_1: np.ndarray,
    kpts_cur: np.ndarray,
    T_world2pixel: np.ndarray,
    match_data_0: MatchDataNP,
):
    T_pixel2world = np.linalg.inv(T_world2pixel)

    # match features

    # 1) match between ref_0 and cur
    model_input_1 = get_model_input_v2(
        torch_utils.to_torch(kpts_ref_0, device=model_ft.device),
        torch_utils.to_torch(kpts_cur, device=model_ft.device),
        torch_utils.to_torch(desc_ref_0, device=model_ft.device),
        torch_utils.to_torch(desc_cur, device=model_ft.device),
    )
    match_data_1 = match_lightglue_finetuned_v2(model_ft, model_input_1)

    # world coords for keypoints
    # 0) match between ref_0 and ref_1
    wpos_kpts_0_0 = depth_utils.pixel_coords_to_world_coords(
        T_pixel2world, depth_ref_0, kpts_ref_0
    )[:, :3]
    wpos_kpts_0_1 = depth_utils.pixel_coords_to_world_coords(
        T_pixel2world, depth_ref_1, kpts_ref_1
    )[:, :3]
    # 1) match between ref_0 and cur
    wpos_kpts_1_0 = wpos_kpts_0_0.copy()
    wpos_kpts_1_1 = depth_utils.pixel_coords_to_world_coords(
        T_pixel2world, depth_cur, kpts_cur
    )[:, :3]

    # map kpts_0_0 to kpts_1_0
    m_0_0_to_1_0 = -np.ones_like(match_data_0.matches_0)
    for idx_0_0 in range(match_data_0.matches_0.shape[0]):
        wpos_0_0 = wpos_kpts_0_0[idx_0_0]
        dists = np.linalg.norm(wpos_0_0 - wpos_kpts_1_0, axis=1)
        idx_1_0 = np.argmin(dists)
        if dists[idx_1_0] < 0.05:
            m_0_0_to_1_0[idx_0_0] = idx_1_0

    # find moving keypoints in match_data_0
    m_moving_0_0_to_0_1 = []
    for idx_0_0 in range(match_data_0.matches_0.shape[0]):
        idx_0_1 = match_data_0.matches_0[idx_0_0]
        if idx_0_1 != -1:
            wpos_0 = wpos_kpts_0_0[idx_0_0]
            wpos_1 = wpos_kpts_0_1[idx_0_1]
            if np.linalg.norm(wpos_0 - wpos_1) > 0.025:
                m_moving_0_0_to_0_1.append((idx_0_0, idx_0_1))
    m_moving_0_0_to_0_1 = np.array(m_moving_0_0_to_0_1)

    # we assume only 1 object is moving between frames
    # compute center of moving points in 0_1
    wcenter_moving_0_1 = np.mean(wpos_kpts_0_1[m_moving_0_0_to_0_1[:, 1]], axis=0)

    # find points in 0_0 close to wcenter_moving_0_1
    closest_points_in_1_0 = predict_utils.get_close_feature_points_np(
        wcenter_moving_0_1, wpos_kpts_1_0, ignore=None, n_closest=10
    )

    # create map for moving keypoints in 0_1 to 1_1
    m_moving_0_1_to_1_1 = []
    for i in range(m_moving_0_0_to_0_1.shape[0]):
        idx_0_0 = m_moving_0_0_to_0_1[i][0]
        idx_0_1 = m_moving_0_0_to_0_1[i][1]
        idx_1_0 = m_0_0_to_1_0[idx_0_0]
        if idx_1_0 != -1:
            idx_1_1 = match_data_1.matches_0[idx_1_0]
            if idx_1_1 != -1:
                m_moving_0_1_to_1_1.append((idx_0_1, idx_1_1))
    m_moving_0_1_to_1_1 = np.array(m_moving_0_1_to_1_1)

    # create map for closest keypoints in 1_1 to 2_1
    m_closest_1_0_to_1_1 = []
    for i in range(closest_points_in_1_0.shape[0]):
        idx_1_0 = closest_points_in_1_0[i]
        idx_1_1 = match_data_1.matches_0[idx_1_0]
        if idx_1_1 != -1:
            m_closest_1_0_to_1_1.append((idx_1_0, idx_1_1))
    m_closest_1_0_to_1_1 = np.array(m_closest_1_0_to_1_1)

    # compute transformation of closest point matches
    # R_closest, t_closest, t_no_rot_closest, x_center, predict_closest = predict_utils.compute_T(
    #     wpos_kpts_1_0[m_closest_1_0_to_1_1[:,0]],
    #     wpos_kpts_1_1[m_closest_1_0_to_1_1[:,1]]
    # )
    t_closest, predict_closest = predict_utils.estimate_translation(
        wpos_kpts_1_0[m_closest_1_0_to_1_1[:, 0]],
        wpos_kpts_1_1[m_closest_1_0_to_1_1[:, 1]],
    )

    # compute new positions of moving points in 2_1
    kpts_2_0 = match_data_1.keypoints_1[m_moving_0_1_to_1_1[:, 1]]
    wpos_kpts_2_0 = wpos_kpts_1_1[m_moving_0_1_to_1_1[:, 1]]
    wpos_kpts_2_1 = predict_closest(wpos_kpts_0_1[m_moving_0_1_to_1_1[:, 0]])
    kpts_2_1 = depth_utils.world_coords_to_pixel_coords(T_world2pixel, wpos_kpts_2_1)
    # kpts_3_2 = env_utils.world_coords_to_pixel_coords(point_cloud_sensor, wpos_kpts_2_2[m_moving_1_2_to_2_2[:,1]])

    return {
        "T": {
            # "R_closest": R_closest,
            "t_closest": t_closest,
            # "t_no_rot_closest": t_no_rot_closest,
            # "x_center": x_center,
        },
        "kpts": {
            "kpts_2_0": kpts_2_0,
            "kpts_2_1": kpts_2_1,
            "wpos_kpts_2_0": wpos_kpts_2_0,
            "wpos_kpts_2_1": wpos_kpts_2_1,
        },
    }


def predict_feature_movement_v3(
    wpos_kpts_0_0: np.ndarray,
    wpos_kpts_0_1: np.ndarray,
    wpos_kpts_1_1: np.ndarray,
    matches_0_0: np.ndarray,
    matches_1_0: np.ndarray,
):
    """
    Args:
        - wpos_kpts_0_0 (np.ndarray): (N, 3) world coordinates of keypoints in ref_0
        - wpos_kpts_0_1 (np.ndarray): (N, 3) world coordinates of keypoints in ref_1
        - wpos_kpts_1_1 (np.ndarray): (N, 3) world coordinates of keypoints in cur
        - matches_0_0 (np.ndarray): (N,) keypoint i in ref_0 matches keypoint matches_0_0[i] in ref_1
        - matches_1_0 (np.ndarray): (N,) keypoint i in ref_0 matches keypoint matches_1_0[i] in cur
    Returns:
        - T (dict): transformation matrix from ref_0 to cur
        - kpts (dict): keypoints in ref_0 and cur
    """

    wpos_kpts_1_0 = wpos_kpts_0_0.copy()

    # # map kpts_0_0 to kpts_1_0
    # m_0_0_to_1_0 = -np.ones_like(matches_0_0)
    # for idx_0_0 in range(matches_0_0.shape[0]):
    #     wpos_0_0 = wpos_kpts_0_0[idx_0_0]
    #     dists = np.linalg.norm(wpos_0_0 - wpos_kpts_1_0, axis=1)
    #     idx_1_0 = np.argmin(dists)
    #     if dists[idx_1_0] < 0.05:
    #         m_0_0_to_1_0[idx_0_0] = idx_1_0

    # find moving keypoints in match_data_0
    m_moving_0_0_to_0_1 = []
    for idx_0_0 in range(matches_0_0.shape[0]):
        idx_0_1 = matches_0_0[idx_0_0]
        if idx_0_1 != -1:
            wpos_0 = wpos_kpts_0_0[idx_0_0]
            wpos_1 = wpos_kpts_0_1[idx_0_1]
            if np.linalg.norm(wpos_0 - wpos_1) > 0.025:
                m_moving_0_0_to_0_1.append((idx_0_0, idx_0_1))
    m_moving_0_0_to_0_1 = np.array(m_moving_0_0_to_0_1)

    # we assume only 1 object is moving between frames
    # compute center of moving points in 0_1
    wcenter_moving_0_1 = np.mean(wpos_kpts_0_1[m_moving_0_0_to_0_1[:, 1]], axis=0)

    # find points in 0_0 close to wcenter_moving_0_1
    # NOTE only consider points that are not moving between ref_0 and ref_1
    ignore = np.zeros(wpos_kpts_1_0.shape[0], dtype=bool)
    for i in range(m_moving_0_0_to_0_1.shape[0]):
        idx_0_0 = m_moving_0_0_to_0_1[i][0]
        idx_1_0 = idx_0_0 # m_0_0_to_1_0[idx_0_0]
        # if idx_1_0 != -1:
        ignore[idx_1_0] = True
    if np.sum(ignore) == ignore.shape[0]:
        # if no points to consider, throw Error
        raise ValueError("No points to consider")
    closest_points_in_1_0 = predict_utils.get_close_feature_points_np(
        wcenter_moving_0_1, wpos_kpts_1_0, ignore=ignore, n_closest=10
    )
    closest_points_in_1_0_without_ignored = []
    for i in range(closest_points_in_1_0.shape[0]):
        idx_1_0 = closest_points_in_1_0[i]
        if not ignore[idx_1_0]:
            closest_points_in_1_0_without_ignored.append(idx_1_0)
    closest_points_in_1_0 = np.array(closest_points_in_1_0_without_ignored)

    # create map for moving keypoints in 0_1 to 1_1
    m_moving_0_1_to_1_1 = []
    for i in range(m_moving_0_0_to_0_1.shape[0]):
        idx_0_0 = m_moving_0_0_to_0_1[i][0]
        idx_0_1 = m_moving_0_0_to_0_1[i][1]
        idx_1_0 = idx_0_0 # m_0_0_to_1_0[idx_0_0]
        # if idx_1_0 != -1:
        idx_1_1 = matches_1_0[idx_1_0]
        if idx_1_1 != -1:
            m_moving_0_1_to_1_1.append((idx_0_1, idx_1_1))
    m_moving_0_1_to_1_1 = np.array(m_moving_0_1_to_1_1)


    # create map for closest keypoints in 1_0 to 1_1
    m_closest_1_0_to_1_1 = []
    for i in range(closest_points_in_1_0.shape[0]):
        idx_1_0 = closest_points_in_1_0[i]
        idx_1_1 = matches_1_0[idx_1_0]
        if idx_1_1 != -1:
            m_closest_1_0_to_1_1.append((idx_1_0, idx_1_1))
    m_closest_1_0_to_1_1 = np.array(m_closest_1_0_to_1_1)

    # compute transformation of closest point matches
    t_closest, predict_closest = predict_utils.estimate_translation(
        wpos_kpts_1_0[m_closest_1_0_to_1_1[:, 0]],
        wpos_kpts_1_1[m_closest_1_0_to_1_1[:, 1]],
    )

    # TODO this should not be part of this function
    # compute new positions of moving points in 2_1
    # kpts_2_0 = kpts_1_1[m_moving_0_1_to_1_1[:,1]]
    wpos_kpts_2_0 = wpos_kpts_1_1[m_moving_0_1_to_1_1[:, 1]]
    wpos_kpts_2_1 = predict_closest(wpos_kpts_0_1[m_moving_0_1_to_1_1[:, 0]])
    # kpts_2_1 = depth_utils.world_coords_to_pixel_coords(T_world2pixel, wpos_kpts_2_1)

    return {
        "T": {
            "t_closest": t_closest,
        },
        "kpts": {
            # "kpts_2_0": kpts_2_0,
            # "kpts_2_1": kpts_2_1,
            "wpos_kpts_2_0": wpos_kpts_2_0,
            "wpos_kpts_2_1": wpos_kpts_2_1,
        },
        "debug": {
            # "m_0_0_to_1_0": m_0_0_to_1_0,
            "m_moving_0_0_to_0_1": m_moving_0_0_to_0_1,
            "wcenter_moving_0_1": wcenter_moving_0_1,
            "closest_points_in_1_0": closest_points_in_1_0,
            "m_moving_0_1_to_1_1": m_moving_0_1_to_1_1,
            "m_closest_1_0_to_1_1": m_closest_1_0_to_1_1,
        },
    }


def compute_img_features(
    model_fts_extract, img: np.ndarray, depth: np.ndarray, T_pixel2world: np.ndarray
):
    """
    Args:
        model_fts_extract (nn.Module): feature extractor model
        img (np.ndarray): (H, W, 3) RGB image
        depth (np.ndarray): (H, W) depth image
        T_pixel2world (np.ndarray): (4, 4) transformation matrix from pixel coordinates to world coordinates
    Returns:
        kpts (np.ndarray): (N, 2) keypoints
        desc (np.ndarray): (N, 256) descriptors
        kpts_scores (np.ndarray): (N,) keypoint scores
    """

    # compute features
    fts = slam_utils.compute_features(model_fts_extract, img)
    fts = torch_utils.to_numpy(fts)
    kpts = fts["keypoints"]
    desc = fts["descriptors"]
    kpts_scores = fts["keypoint_scores"]

    # Only consider features that are inside the table region
    wpos_kpts = depth_utils.pixel_coords_to_world_coords(
        T_pixel2world=T_pixel2world, real_depth=depth, pixel_coords_xy=kpts
    )
    mask = match_utils.create_feature_table_mask(wpos_kpts)
    kpts = kpts[mask]
    wpos_kpts = wpos_kpts[mask]
    desc = desc[mask]
    kpts_scores = kpts_scores[mask]

    return kpts, desc, kpts_scores, wpos_kpts


def compute_matches(
    model_fts_extract,
    model_fts_match,
    img_0: np.ndarray,
    img_1: np.ndarray,
    depth_0: np.ndarray,
    depth_1: np.ndarray,
    T_pixel2world: np.ndarray,
):
    kpts_0, desc_0, kpts_scores_0, wpos_kpts_0 = compute_img_features(
        model_fts_extract, img_0, depth_0, T_pixel2world
    )
    kpts_1, desc_1, kpts_scores_1, wpos_kpts_1 = compute_img_features(
        model_fts_extract, img_1, depth_1, T_pixel2world
    )

    # match between img_0 and img_1
    model_input_0 = get_model_input_v3_np(kpts_0, kpts_1, desc_0, desc_1)
    match_data_0 = match_lightglue_finetuned_v2(model_fts_match, model_input_0)
    match_data = {
        "img_0": img_0,
        "img_1": img_1,
        "depth_0": depth_0,
        "depth_1": depth_1,
        "kpts_0": kpts_0,
        "kpts_1": kpts_1,
        "wpos_kpts_0": wpos_kpts_0,
        "wpos_kpts_1": wpos_kpts_1,
        "desc_0": desc_0,
        "desc_1": desc_1,
        "kpts_scores_0": kpts_scores_0,
        "kpts_scores_1": kpts_scores_1,
        "matches_0": match_data_0.matches_0,
        "matches_1": match_data_0.matches_1,
    }
    return torch_utils.to_numpy(match_data)


def predict_feature_movement_from_ref_data(
    model_fts_extract,
    model_fts_match,
    T_world2pixel: np.ndarray,
    img_cur: np.ndarray,
    depth_cur: np.ndarray,
    ref_match_data,
):
    T_pixel2world = np.linalg.inv(T_world2pixel)

    # compute features of cur image
    kpts_cur, desc_cur, kpts_scores_cur, wpos_kpts_cur = compute_img_features(
        model_fts_extract, img_cur, depth_cur, T_pixel2world
    )

    # match between ref_0 image and cur image
    model_input_1 = get_model_input_v3_np(
        ref_match_data["kpts_0"], kpts_cur, ref_match_data["desc_0"], desc_cur
    )
    match_data_1 = match_lightglue_finetuned_v2(model_fts_match, model_input_1)
    match_data_1 = torch_utils.to_numpy(match_data_1)

    # predict feature movement
    pred_res = predict_feature_movement_v3(
        wpos_kpts_0_0=ref_match_data["wpos_kpts_0"],
        wpos_kpts_0_1=ref_match_data["wpos_kpts_1"],
        wpos_kpts_1_1=wpos_kpts_cur,
        matches_0_0=ref_match_data["matches_0"],
        matches_1_0=match_data_1.matches_0,
    )

    # convert world coordinates to pixel coordinates
    kpts_2_0 = depth_utils.world_coords_to_pixel_coords(
        T_world2pixel, pred_res["kpts"]["wpos_kpts_2_0"]
    )
    kpts_2_1 = depth_utils.world_coords_to_pixel_coords(
        T_world2pixel, pred_res["kpts"]["wpos_kpts_2_1"]
    )
    pred_res["kpts"]["kpts_2_0"] = kpts_2_0
    pred_res["kpts"]["kpts_2_1"] = kpts_2_1
    pred_res["debug"]["kpts_cur"] = kpts_cur
    pred_res["debug"]["desc_cur"] = desc_cur
    pred_res["debug"]["wpos_kpts_cur"] = wpos_kpts_cur

    return pred_res


def visualize_keypoint_matches_0(img_0, img_1, kpts_0, kpts_1, matches_0):
    kpts_0 = kpts_0[matches_0 != -1]
    kpts_1 = kpts_1[matches_0[matches_0 != -1]]
    show_matches_kpts(img_0, img_1, kpts_0, kpts_1) 


def visualize_keypoint_mapping(img_0, img_1, kpts_0, kpts_1, kpts_m):
    matches = -np.ones(kpts_0.shape[0], dtype=int)
    matches[kpts_m[:, 0]] = kpts_m[:, 1]
    visualize_keypoint_matches_0(img_0, img_1, kpts_0, kpts_1, matches)
