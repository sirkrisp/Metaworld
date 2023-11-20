from scipy.interpolate import RegularGridInterpolator
import numpy as np


def pixel_coords_to_world_coords(T_pixel2world, real_depth, pixel_coords_xy = None):
    """
    Convert image coordinates to world coordinates.

    Parameters:
    - T_pixel2world
    - real_depth: np.ndarray with shape (h, w), representing the depth map.
    - pixel_coords_xy: np.ndarray with shape (n, 2), representing the n image coordinates.
    """

    if pixel_coords_xy is None:
        x = np.linspace(0, real_depth.shape[0] - 1, real_depth.shape[0])
        y = np.linspace(0, real_depth.shape[1] - 1, real_depth.shape[1])
        pixel_coords_xy = np.vstack(np.meshgrid(y, x)).reshape(2, -1).T

    x = np.linspace(0, real_depth.shape[0] - 1, real_depth.shape[0])
    y = np.linspace(0, real_depth.shape[1] - 1, real_depth.shape[1])
    # TODO instead of interpolation find minimum across neighours
    # Or Alternatively take 
    interp_depth = RegularGridInterpolator((x, y), real_depth, bounds_error=False, fill_value=100000)

    pixel_coords_xy_center = np.floor(pixel_coords_xy)
    offsets = np.array([[0,0], [0,1], [1,0], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]])
    pixel_coords_xy_region = pixel_coords_xy_center[:, np.newaxis, :] + offsets[np.newaxis, :, :] # shape (n, 9, 2)

    pixel_z_coords = np.ones((pixel_coords_xy.shape[0], 1), dtype=np.float32) * 100000
    for i in range(9):
        pixel_z_coords_i = interp_depth(pixel_coords_xy_region[:, i, :] @ np.array([[0, 1], [1, 0]]))[:, np.newaxis]  # shape (n, 1)
        # take min depth across all 9 neighbours
        pixel_z_coords = np.minimum(pixel_z_coords, pixel_z_coords_i)  # shape (n, 1)
    # pixel_z_coords /= 9.0

    # pixel_z_coords = interp_depth(pixel_coords_xy @ np.array([[0, 1], [1, 0]]))[:, np.newaxis]  # shape (n, 1)

    pixel_coords = np.hstack((pixel_coords_xy * pixel_z_coords, pixel_z_coords, np.ones_like(pixel_z_coords)))
    world_coords = (T_pixel2world @ pixel_coords.T).T
    
    return world_coords, pixel_coords