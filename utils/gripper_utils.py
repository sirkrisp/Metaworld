import numpy as np
from typing import Callable
import utils.geom_utils as geom_utils


def shift_gripper_points(
    potential_gripper_points: np.ndarray,
    gripper_normal: np.ndarray,
    gripper_offset: float,
):
    """
    Args:
        potential_gripper_points (np.ndarray): (N, 3)
        gripper_normal (np.ndarray): (3,) normal vector of gripper (pointing away from surface)
        gripper_offset (float): offset in the direction of gripper_normal
    Returns:
        potential_gripper_points (np.ndarray): (N, 3)
    """
    return potential_gripper_points + gripper_normal * gripper_offset
    


def check_gripper_has_collision(
    potential_gripper_points: np.ndarray,
    gripper_normal: np.ndarray,
    is_occupied: Callable[[np.ndarray], np.ndarray],
):
    """
    Args:
        potential_gripper_points (np.ndarray): (N, 3)
        gripper_offset np.ndarray (3,): gripper is positioned at potential_gripper_points[i] + gripper_offset
        is_occupied (Callable[[np.ndarray], float]): function that returns occupancy value at a given point
    """
    n_points_per_dim = 10
    n_sample_points = (
        n_points_per_dim * n_points_per_dim
    )  # gripper_points_sampler.get_num_sample_points()

    n_contact_points = potential_gripper_points.shape[0]
    sample_points = np.zeros((n_sample_points * n_contact_points, 3))

    # sample points around potential_gripper_points orthogonal to gripper_normal
    gripper_pad_width = 0.03
    # gripper_depth = 0.006

    for i in range(n_contact_points):
        sample_points[
            i * n_sample_points : (i + 1) * n_sample_points
        ] = geom_utils.sample_plane_grid(
            potential_gripper_points[i],
            gripper_normal,
            gripper_pad_width,
            n_points_per_dim,
        )

    # check if sampled points are occupied
    pred_is_occupied = is_occupied(sample_points)

    has_collision = np.zeros(n_contact_points, dtype=bool)
    for i in range(n_contact_points):
        has_collision[i] = np.any(
            pred_is_occupied[i * n_sample_points : (i + 1) * n_sample_points]
        )

    return has_collision, pred_is_occupied, sample_points