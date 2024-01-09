from scipy.interpolate import RegularGridInterpolator
import numpy as np


def pixel_coords_to_world_coords_simple(T_pixel2world, real_depth, pixel_coords_xy = None, return_pixel_coords_xy=False):
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
    interp_depth = RegularGridInterpolator((x, y), real_depth, bounds_error=False, fill_value=100000)

    pixel_z_coords = interp_depth(pixel_coords_xy @ np.array([[0, 1], [1, 0]]))[:, np.newaxis]  # shape (n, 1)
    pixel_coords = np.hstack((pixel_coords_xy * pixel_z_coords, pixel_z_coords, np.ones_like(pixel_z_coords)))
    world_coords = (T_pixel2world @ pixel_coords.T).T

    if return_pixel_coords_xy:
        return world_coords, pixel_coords_xy
    
    return world_coords


def pixel_coords_to_world_coords(T_pixel2world: np.ndarray, real_depth: np.ndarray, pixel_coords_xy: np.ndarray = None, return_adjusted_pixel_coords_xy=False):
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

    adjusted_pixel_coords_xy = pixel_coords_xy.copy()

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
        is_minimum_i = pixel_z_coords_i < pixel_z_coords
        adjusted_pixel_coords_xy[is_minimum_i[:,0], :] = pixel_coords_xy_region[:, i, :][is_minimum_i[:,0], :]
        pixel_z_coords = np.minimum(pixel_z_coords, pixel_z_coords_i)  # shape (n, 1)
    # pixel_z_coords /= 9.0

    # pixel_z_coords = interp_depth(pixel_coords_xy @ np.array([[0, 1], [1, 0]]))[:, np.newaxis]  # shape (n, 1)

    pixel_coords = np.hstack((adjusted_pixel_coords_xy * pixel_z_coords, pixel_z_coords, np.ones_like(pixel_z_coords)))
    world_coords = (T_pixel2world @ pixel_coords.T).T

    if return_adjusted_pixel_coords_xy:
        return world_coords, pixel_coords, adjusted_pixel_coords_xy
    
    return world_coords

def pixel_coords_to_world_coords_moving_pixels(T_pixel2world, real_depth_0, real_depth_1, pixel_coords_xy = None):
    """
    Convert image coordinates to world coordinates.

    Parameters:
    - T_pixel2world
    - real_depth: np.ndarray with shape (h, w), representing the depth map.
    - pixel_coords_xy: np.ndarray with shape (n, 2), representing the n image coordinates.
    """


    if pixel_coords_xy is None:
        x = np.linspace(0, real_depth_0.shape[0] - 1, real_depth_0.shape[0])
        y = np.linspace(0, real_depth_0.shape[1] - 1, real_depth_0.shape[1])
        pixel_coords_xy = np.vstack(np.meshgrid(y, x)).reshape(2, -1).T

    adjusted_pixel_coords_xy = pixel_coords_xy.copy()

    x = np.linspace(0, real_depth_0.shape[0] - 1, real_depth_0.shape[0])
    y = np.linspace(0, real_depth_0.shape[1] - 1, real_depth_0.shape[1])
    # TODO instead of interpolation find minimum across neighours
    # Or Alternatively take 
    interp_depth_0 = RegularGridInterpolator((x, y), real_depth_0, bounds_error=False, fill_value=100000)
    interp_depth_1 = RegularGridInterpolator((x, y), real_depth_1, bounds_error=False, fill_value=100000)

    pixel_coords_xy_center = np.floor(pixel_coords_xy)
    offsets = np.array([[0,0], [0,1], [1,0], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]])
    pixel_coords_xy_region = pixel_coords_xy_center[:, np.newaxis, :] + offsets[np.newaxis, :, :] # shape (n, 9, 2)

    moving_pixels = np.zeros((pixel_coords_xy.shape[0], 1), dtype=bool)

    pixel_z_coords = np.ones((pixel_coords_xy.shape[0], 1), dtype=np.float32) * 100000
    for i in range(9):
        pixel_z_coords_i_0 = interp_depth_0(pixel_coords_xy_region[:, i, :] @ np.array([[0, 1], [1, 0]]))[:, np.newaxis]  # shape (n, 1)
        pixel_z_coords_i_1 = interp_depth_1(pixel_coords_xy_region[:, i, :] @ np.array([[0, 1], [1, 0]]))[:, np.newaxis]  # shape (n, 1)
        moving_pixels_i = np.abs(pixel_z_coords_i_0 - pixel_z_coords_i_1) > 0.001 # shape (n, 1)
        moving_pixels = np.logical_or(moving_pixels, moving_pixels_i)
        adjusted_pixel_coords_xy[moving_pixels_i[:,0], :] = pixel_coords_xy_region[:, i, :][moving_pixels_i[:,0], :]
        # take min depth across all 9 neighbours
        # TODO also needs to be adjusted
        pixel_z_coords = np.minimum(pixel_z_coords, pixel_z_coords_i_0)  # shape (n, 1)
        pixel_z_coords[moving_pixels_i] = pixel_z_coords_i_0[moving_pixels_i]
    # pixel_z_coords /= 9.0

    # pixel_z_coords = interp_depth(pixel_coords_xy @ np.array([[0, 1], [1, 0]]))[:, np.newaxis]  # shape (n, 1)

    # TODO rename pixel_coords
    pixel_coords = np.hstack((adjusted_pixel_coords_xy * pixel_z_coords, pixel_z_coords, np.ones_like(pixel_z_coords)))
    world_coords = (T_pixel2world @ pixel_coords.T).T
    
    return world_coords, pixel_coords, moving_pixels[:,0], adjusted_pixel_coords_xy


def world_coords_to_pixel_coords(T_world2pixel: np.ndarray, world_coords: np.ndarray):
    """
    Convert world coordinates to image coordinates.

    Parameters:
    - point_cloud_sensor: PointCloudSensor object.
    - world_coords: np.ndarray with shape (n, 3), representing the n world coordinates.
    """
    if world_coords.shape[1] == 3:
        world_coords = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))
    pixel_coords = (T_world2pixel @ world_coords.T).T
    pixel_coords_xy = pixel_coords[:, :2] / pixel_coords[:, 2:3]
    return pixel_coords_xy