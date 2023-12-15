

import numpy as np
import pyquaternion as pq

def ray_plane_intersection(p_o : np.ndarray, p_n: np.ndarray, r_o : np.ndarray, r_dir : np.ndarray):
    """
    solve for t:
        p = ray_o + t * ray_dir
        (p-p_o).p_n = 0
        => t = (p_o - r_o).p_n / (r_dir.p_n) 
    """
    r_dir_p_n = np.dot(r_dir, p_n)
    if np.abs(r_dir_p_n) < 1e-8:
        # parallel
        return False
    t = np.dot(p_o - r_o, p_n) / r_dir_p_n
    return r_o + t * r_dir

def camera_lookat_world(q):
    quaternion = pq.Quaternion(np.array(q))
    lookat_world = quaternion.rotate(np.array([0,0,1]))
    return lookat_world


def camera_lookat_world_v2(camera):
    quaternion = camera.quaternion
    quaternion = pq.Quaternion(quaternion[3], quaternion[0], quaternion[1], quaternion[2])
    lookat_world = quaternion.rotate(np.array([0, 0, 1]))
    return lookat_world

def get_focal_length(fov, width, height, film_width = 0.035):
    """
    Args:
        - fov: vertical angle
        - width: width in pixels
        - height: height in pixels
    """
    aspect = width / height

    film_height = film_width / aspect
    focal = 0.5 * film_height / np.tan(0.5 * fov)
    return focal

def mouse2point(mx, my, renderer, camera):
    film_width = 0.035  # default threejs
    width = renderer.width
    height = renderer.height
    film_height = film_width / camera.aspect  # default threejs
    fov_radians = camera.fov / 360 * np.pi * 2
    focal = 0.5 * film_height / np.tan(0.5 * fov_radians)
    # print(mx / width - 0.5, -my / height + 0.5)
    xp = (mx / width - 0.5) * film_width
    yp = -(my / height - 0.5) * film_height
    zp = -focal
    return np.array([xp, yp, zp])


def mouse_plane_intersection(mx, my, plane_origin, plane_normal, renderer, camera):
    pt_on_sensor = mouse2point(mx, my, renderer, camera)
    ray_origin = np.array(camera.position)
    quaternion = camera.quaternion
    quaternion = pq.Quaternion(quaternion[3], quaternion[0], quaternion[1], quaternion[2])
    ray_dir = quaternion.rotate(pt_on_sensor)
    ray_dir /= np.linalg.norm(ray_dir)
    pt_on_plane = ray_plane_intersection(plane_origin, plane_normal, ray_origin, ray_dir)
    return pt_on_plane
