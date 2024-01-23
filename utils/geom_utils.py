""" Geometry utilities """
import numpy as np


def sample_xz_plane_grid(center_pt, grid_size, n_points_per_dim):
    """Sample grid of points in xz plane
    Args:
        center_pt: np.ndarray (3)
        grid_size: float
        n_points_per_dim: int
    """
    x = np.linspace(
        center_pt[0] - grid_size / 2, center_pt[0] + grid_size / 2, n_points_per_dim
    )
    z = np.linspace(
        center_pt[2] - grid_size / 2, center_pt[2] + grid_size / 2, n_points_per_dim
    )
    x, z = np.meshgrid(x, z)
    y = np.ones_like(x) * center_pt[1]
    grid = np.stack([x, y, z], axis=-1)
    return grid.reshape(-1, 3)


def sample_plane_grid(plane_center, plane_normal, grid_size, n_points_per_dim):
    """Sample grid of points on a plane
    Args:
        plane_center: np.ndarray (3)
        plane_normal: np.ndarray (3)
        grid_size: float
        n_points_per_dim: int
    Returns:
        grid: np.ndarray (N, 3) - grid of points on the plane
    """

    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Generate a set of orthogonal vectors to the plane normal
    # v1 = np.array([plane_normal[1], -plane_normal[0], 0])
    if plane_normal[0] == 0 and plane_normal[1] == 0:
        v1 = np.array([1, 0, 0])
    else:
        v1 = np.array([plane_normal[1], -plane_normal[0], 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(plane_normal, v1)

    # Normalize the orthogonal vectors
    v2 = v2 / np.linalg.norm(v2)

    # Generate grid points
    x = np.linspace(
        -grid_size / 2, + grid_size / 2, n_points_per_dim
    )
    y = np.linspace(
        -grid_size / 2, + grid_size / 2, n_points_per_dim
    )
    x, y = np.meshgrid(x, y)

    # Transform grid points to the plane
    grid = plane_center + x[:, :, np.newaxis] * v1 + y[:, :, np.newaxis] * v2

    return grid.reshape(-1, 3)


def calculate_angle(a: np.ndarray, b: np.ndarray):
    """ Calculate angle between two vectors
    Args:
        a: (N, 3) np.ndarray
        b: (3) np.ndarray
    Returns:
        angle_degrees (np.ndarraz): (N,) angle degrees
    """
    # Normalize vectors
    a_normalized = np.linalg.norm(a, axis=1)
    b_normalized = np.linalg.norm(b)

    # Calculate dot product
    dot_product = np.dot(a, b)

    # Calculate cosine similarity
    cosine_similarity = dot_product / (a_normalized * b_normalized)

    # Calculate angle in radians
    angle_radians = np.arccos(cosine_similarity)

    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees