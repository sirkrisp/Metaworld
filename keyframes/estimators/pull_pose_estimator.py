from matplotlib.pyplot import step
import numpy as np
import utils.geom_utils as geom_utils
from typing import Callable, List
import utils.gripper_utils as gripper_utils
import keyframes.seg_any as seg_any
import keyframes.mcc as mcc
import utils.torch_utils as torch_utils
import copy


def generate_potential_pull_poses_for_surface(
    surface_verts: np.ndarray,
    surface_vert_normals: np.ndarray,
    is_occupied: Callable[[np.ndarray], np.ndarray],
    pull_direction: np.ndarray,
):
    """
    Args:
        surface_verts: (N, 3)
        surface_vert_normals: (N, 3)
        is_occupied: Callable[[np.ndarray], np.ndarray]
        pull_direction: (3,)
    Returns:
        potential_push_positions: (M, 3)
    """

    pull_direction = pull_direction / np.linalg.norm(pull_direction)

    # 1) only consider vertices that are aligned with pull_direction
    angle_err = 20
    while True:
        angle_degrees = geom_utils.calculate_angle(surface_vert_normals, -pull_direction)
        angle_mask = angle_degrees < angle_err
        if np.sum(angle_mask) > 0:
            break
        angle_err += 1
    potential_push_positions = surface_verts[angle_mask]

    # 2) collision checking
    gripper_offset = 0.001
    max_tries = 100
    gripper_normal = -pull_direction
    for i in range(max_tries):

        # shift gripper points in the direction of push_direction
        potential_push_positions = gripper_utils.shift_gripper_points(
            potential_push_positions, gripper_normal, gripper_offset
        )
        
        has_collision, _, _ = gripper_utils.check_gripper_has_collision(
            potential_push_positions,
            gripper_normal=gripper_normal,
            is_occupied=is_occupied,
        )

        if np.sum(~has_collision) == 0:
            if i == max_tries - 1:
                raise RuntimeError("Could not find gripper points that do not collide")
            continue

        potential_push_positions = potential_push_positions[~has_collision, :]
        break

    return potential_push_positions


def select_best_push_pos(
        surface_verts: np.ndarray, 
        potential_push_positions: np.ndarray, 
        push_direction: np.ndarray
):
    """
    Args:
        surface_verts: (N, 3)
        potential_push_positions: (M, 3)
        push_direction: (3,)
    Returns:
        push_pos: (3,)
    """
    push_direction = push_direction / np.linalg.norm(push_direction)
    
    # Generate a set of orthogonal vectors to the plane normal
    # v1 = np.array([plane_normal[1], -plane_normal[0], 0])
    if push_direction[0] == 0 and push_direction[1] == 0:
        v1 = np.array([1, 0, 0])
    else:
        v1 = np.array([push_direction[1], -push_direction[0], 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(push_direction, v1)
    v2 = v2 / np.linalg.norm(v2)

    # select top 10% of points that are closest to the push_direction
    surface_center = np.mean(surface_verts, axis=0)
    potential_push_positions_cyl_coords = potential_push_positions - surface_center
    potential_push_positions_cyl_coords = np.dot(potential_push_positions_cyl_coords, np.stack([v1, v2], axis=1))
    dists = np.linalg.norm(potential_push_positions_cyl_coords, axis=1)
    n_points = potential_push_positions.shape[0]
    n_points_to_select = int(np.ceil(n_points * 0.1))
    idx = np.argsort(dists)[:n_points_to_select]
    potential_push_positions = potential_push_positions[idx]

    # select point that is frontmost oppsite to push_direction
    dot_prod = np.dot(potential_push_positions, -push_direction)
    idx = np.argmax(dot_prod)
    push_pos = potential_push_positions[idx]
    
    return push_pos



class PushPoseEstimator:

    
    def __init__(self, mcc_reconstructor: mcc.ReconstructMCC, seg_any_estimator: seg_any.SegAny, device="cuda"):
        self.mcc_reconstructor = mcc_reconstructor
        self.seg_any_predictor = seg_any_estimator
        self.step_data = {}
        self.device = device

    def predict_push_pose(self,**args):
        self.predict_push_pose_stepwise(0, **args)
        self.predict_push_pose_stepwise(1, **args)
        return self.predict_push_pose_stepwise(2, **args)

    def predict_push_pose_stepwise(
            self, 
            step: int, 
            img: np.ndarray, 
            xyz: np.ndarray, 
            kpts_2_0: np.ndarray,
            wpos_kpts_2_0: np.ndarray, 
            wpos_kpts_2_1: np.ndarray, 
            mcc_params = {"grid_granularity": 0.025}
    ):
        """
        Args:
            step: int
            img: (3, H, W) np.ndarray
            xyz: (H*W, 3) np.ndarray
            kpts: (N, 2) np.ndarray
            push_direction: (3,) np.ndarray
        Returns:
            push_pose: (3,) np.ndarray
        """

        if step == 0:
            self.step_data = {}
            self.step_data["seg"] = self.seg_any_predictor.compute_mask(img, kpts_2_0)
        elif step == 1:
            self.mcc_reconstructor.set_model_input(
                img,
                self.step_data["seg"],
                xyz,
                grid_granularity=mcc_params["grid_granularity"],
            )
            (
                self.step_data["pred_occupy"],
                self.step_data["pred_colors"],
                self.step_data["occupy_threshold"],
                self.step_data["verts"],
                self.step_data["faces"],
                self.step_data["vert_normals"],
                self.step_data["values"],
            ) = self.mcc_reconstructor.reconstruct()
        elif step == 2:
            occupy_threshold = self.step_data["occupy_threshold"]

            def is_occupied(query_points: np.ndarray):
                query_points_normalized = self.mcc_reconstructor.normalize(query_points)
                query_points_normalized = torch_utils.to_torch(query_points_normalized, self.device)
                pred_occupied, _ = self.mcc_reconstructor.query_model(query_points_normalized)
                pred_occupied = torch_utils.to_numpy(pred_occupied)
                return pred_occupied > occupy_threshold
            
            verts_world = self.mcc_reconstructor.denormalize(self.step_data["verts"])
            surface_vert_normals = -self.step_data["vert_normals"]  # normals are inverted
            push_vector = np.mean(wpos_kpts_2_1 - wpos_kpts_2_0, axis=0)[:3]
            push_direction = push_vector / np.linalg.norm(push_vector)
            possible_push_vectors = np.concatenate([np.eye(3), -np.eye(3)], axis=0)
            angles = geom_utils.calculate_angle(possible_push_vectors, push_direction)
            push_direction = possible_push_vectors[np.argmin(angles)]
            push_vector = np.dot(push_direction, push_vector) * push_direction

            potential_push_positions = generate_potential_push_poses_for_surface(
                verts_world,
                surface_vert_normals,
                is_occupied,
                push_direction=push_direction
            )
            push_start_position = select_best_push_pos(verts_world, potential_push_positions, push_direction)

            self.step_data["push_start_pos"] = push_start_position
            self.step_data["push_goal_pos"] = push_start_position + push_vector

        return copy.deepcopy(self.step_data)