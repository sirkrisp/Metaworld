from typing import List
import torch
import numpy as np
from keyframes.align_ft_spair.utils import ft_align_utils
import pyquaternion as pyq


def get_calibration_matrix(img_h, img_w, f=0.3):
    # between about 40 and 58mm
    # fx = 100 #0.04
    # fy = 100 #fx * img_h / img_w
    # pixel_size = 1 #0.036 / max(img_h,img_w)
    px = img_w / 2 / max(img_w, img_h)
    py = img_h / 2 / max(img_w, img_h)
    K = np.array([
        [f, 0, px, 0],
        [0, f, py, 0],
        [0, 0, 1, 0],
        [0,0,0,1]
    ])
    return K


def get_inv_calibration_matrix(img_h, img_w, f):
    """ Construct inverse of calibration matrix analytically
    """
    px = img_w / 2 / max(img_w, img_h)
    py = img_h / 2 / max(img_w, img_h)
    f_plus = f + 1e-8
    K_inv = torch.Tensor([
        [1/f_plus, 0, -px/f_plus],
        [0, 1/f_plus, -py/f_plus],
        [0, 0, 1]
    ])
    return K_inv



def project_to_img_plane(xyz_world: torch.Tensor, K: torch.Tensor):
    """
    Args:
        xyz_world: (n, 3)
        K: (4, 4) is the calibration matrix 
            (we assume that the camera is located at the origin and looks along the z-axis.
            That is, T_cam = [I|0] and R_cam = I)
    Out:
        xy_img: (n, 2)
    """
    xyz_world = torch.cat([xyz_world, torch.ones(xyz_world.shape[0], 1).double()], dim=1)
    xy_img = torch.matmul(K, xyz_world.T).T
    xy_img = xy_img[:,:2] / (xy_img[:,2][:,None] + 1e-9)
    return xy_img


def reproject_to_world(xy_img: torch.Tensor, depth_values: torch.Tensor, K_inv: torch.Tensor):
    """
    Args:
        xy_img: (n, 2)
        depth_values: (n)
        K_inv: (3, 3) is the inverse of the calibration matrix K[:3,:3]
    Out:
        xyz_world: (n, 3)
    """
    xyz_cam_plane = torch.cat([xy_img, depth_values.unsqueeze(0)], dim=1)
    xyz_world = torch.matmul(K_inv, xyz_cam_plane.T)
    return xyz_world.T


# def get_depth_values(depth_img: torch.Tensor, xy_img: torch.Tensor):
#     """
#     Args:
#         depth_img: (h, w)
#         xy_img: (n, 2)
#     Out:
#         depth_values: (n)
#     """
#     # TODO might be wrong (x,y interchanges)
#     depth_values = torch.nn.functional.grid_sample(depth_img[None,None,:,:].float(), xy_img[:,:,None,None], align_corners=True)
#     depth_values = depth_values[0,0]
#     return depth_values


def reproject_depth(depth_img: np.ndarray, focal_length=0.3):
    """
    Args:
        depth_img: (h, w)
    """
    img_h, img_w = depth_img.shape
    K = get_calibration_matrix(img_h, img_w, f=focal_length)
    # T = np.eye(4)
    # pixel_size = 20*0.036 / max(img_h,img_w)
    x, y = np.meshgrid(np.arange(img_w), np.arange(img_h))
    x = x.astype(float) / max(img_w, img_h)
    y = y.astype(float) / max(img_w, img_h)
    # x *= pixel_size
    # y *= pixel_size
    xyz_cam_plane = np.stack([x * depth_img, y * depth_img, depth_img])  # (3,h,w)
    xyz_cam_plane = xyz_cam_plane.reshape((3, img_h*img_w))
    xyz_world = (np.linalg.inv(K[:3,:3]) @ xyz_cam_plane).T
    xyz_world = xyz_world.reshape((img_h, img_w, 3))
    # print(np.linalg.inv(K[:3,:3]))
    return xyz_world


def extract_point_v2(xyz_img: torch.Tensor, weight: torch.Tensor, n_topk=9):
    """
    Args:
        xyz_img: (h*w, 3) or (h, w, 3)
        weight: (h, w) or (h*w)
    Out:
        v0: (3,)
    """
    h, w = xyz_img.shape[:2]
    weight = weight.reshape(-1)
    sort_res = torch.sort(weight, descending=True)
    weight = weight[sort_res.indices[:n_topk]]
    best_index = sort_res.indices[0]
    best_index_x = best_index % w
    best_index_y = best_index // w

    # TODO replace with convolution
    # get minimum depth by looking around best (x,y)
    v_with_min_z = xyz_img[best_index_y, best_index_x]
    for i in range(-1, 2):
        for j in range(-1, 2):
            if best_index_x + i >= 0 and best_index_x + i < w and best_index_y + j >= 0 and best_index_y + j < h:
                v = xyz_img[best_index_y + j, best_index_x + i]
                if v[2] < v_with_min_z[2]:
                    v_with_min_z = v
    return v_with_min_z


def extract_point(xyz_img: torch.Tensor, weight: torch.Tensor, n_topk=18, threshold_z=0.7):
    """
    Args:
        xyz_img: (h*w, 3) or (h, w, 3)
        weight: (h, w) or (h*w)
    Out:
        v0: (3,)
    """
    xyz_img = xyz_img.reshape((-1, 3))
    weight = weight.reshape(-1)
    sort_res = torch.sort(weight, descending=True)
    weight = weight[sort_res.indices[:n_topk]]
    v0 = xyz_img[sort_res.indices[:n_topk], :]
    mask = v0[:,2] < threshold_z
    v0 = v0[mask,:]
    weight = weight[mask]
    weight /= torch.sum(weight)
    v0 = (v0 * weight[:,None]).sum(dim=0)
    return v0


def construct_local_coord_system(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray):
    """
    Args:
        v1: (3,)
        v2: (3,)
        v3: (3,)
    Out:
        R: (3,3)
        t: (3,)
        det: determinant of (v1, v2, v3)
    """
    v1v2 = v2 - v1
    v1v3 = v3 - v1
    v1v2_norm = np.linalg.norm(v1v2)
    v1v3_norm = np.linalg.norm(v1v3)
    if v1v2_norm < 1e-8 or v1v3_norm < 1e-8:
        return None, None, 0
    x_ax = v1v2 / v1v2_norm
    z_ax = np.cross(v1v2, v1v3 / v1v3_norm)
    z_ax /= np.linalg.norm(z_ax)
    y_ax = np.cross(x_ax, z_ax)
    y_ax /= np.linalg.norm(y_ax)

    # compute determinant of v1,v2,v3
    det = np.linalg.det(np.stack([v1/np.linalg.norm(v1),v2/np.linalg.norm(v2),v3/np.linalg.norm(v3)]))
    t = v1
    R = np.stack([x_ax, y_ax, z_ax])
    return R, t, det


def construct_plane(
    img_embd: torch.Tensor, 
    img_xyz: torch.Tensor,
    kpt_features_avg_train: torch.Tensor,
    stable_indices: List[int],
    oriented_triangles: List[List[int]],
    # stable_threshold = 0.00048,
    stable_threshold_rate = 0.8,
):
    """
    Args:
        img_embd: (C, h, w),
        img_xyz: (h*w, 3)
    """
    _, h, w = img_embd.shape
    kpt_attn = ft_align_utils.compute_attn(kpt_features_avg_train, img_embd[None,:,:,:])  # (n_kpts, h, w)
    kpt_attn_max = torch.max(torch.max(kpt_attn, dim=-1).values, dim=-1).values  # (n_kpts,)
    
    thresh = stable_threshold_rate * torch.max(kpt_attn_max)
    plane_kpt_indices = np.array(stable_indices)[kpt_attn_max[stable_indices] >= thresh]
    plane_kpt_indices_set = set(plane_kpt_indices.tolist())
    # n_kpts = len(plane_kpt_indices)
    
    plane_kpt_xyz = []
    plane_kpt_idx_to_xyz = {}
    for kpt_index in plane_kpt_indices:
        plane_kpt_xyz.append(extract_point_v2(img_xyz, kpt_attn[kpt_index]))
        plane_kpt_idx_to_xyz[kpt_index] = plane_kpt_xyz[-1]

    # plane center = average of plane_kpt_xyz
    plane_origin = torch.zeros(3).double()
    if len(plane_kpt_indices) == 0:
        print("Warning:", "no keypoints found")
        print("kpt_attn_max", kpt_attn_max)
    else:
        plane_origin = torch.mean(torch.stack(plane_kpt_xyz), dim=0)

    # construct oriented triangles
    trianlge_dirs = []
    for triangle in oriented_triangles:
        i1, i2, i3 = triangle
        if all([idx in plane_kpt_indices_set for idx in triangle]):
            # print("using triangle", triangle)
            R, t, det = construct_local_coord_system(
                plane_kpt_idx_to_xyz[i1].numpy(),
                plane_kpt_idx_to_xyz[i2].numpy(),
                plane_kpt_idx_to_xyz[i3].numpy()
            )
            if abs(det) > 0.001:
                trianlge_dirs.append(torch.from_numpy(R[2,:]))
    plane_dir = torch.tensor([0,0,1.0]).double()
    if len(trianlge_dirs) == 0:
        print("Warning:", "triangles have low determinant or no triangle could be constructed")
        print("plane kpt indices", plane_kpt_indices)
    else:
        plane_dir = torch.mean(torch.stack(trianlge_dirs), dim=0)
    return plane_origin, plane_dir, plane_kpt_xyz


def create_grid(center, normal, n_cells, cell_size=0.01):
    """
    Args:
        center: (3,)
        normal: (3,)
        n_cells: int
    Out:
        grid: (n_cells, n_cells, 3)
    """
    # normal . x_axis = 0
    # => x_axis = [1/normal[0], 1/normal[1], -1/normal[2]]
    # normal . y_axis = 0
    # rotate normal
    axis = [0,0,1] if abs(normal[1]) < 1e-3 and abs(normal[2]) < 1e-3 else [1,0,0]
    q0 = pyq.Quaternion(axis=axis, angle=90.0) # Rotate 0 about x=y=z
    normal /= np.linalg.norm(normal)
    normal_rotated = np.array(q0.rotate(normal))
    x_axis = np.cross(normal_rotated, normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Create a grid of points
    grid = np.zeros((n_cells, n_cells, 3))
    for i in range(n_cells):
        for j in range(n_cells):
            # Calculate the position of each point on the grid
            grid[i, j] = center + (i - n_cells // 2) * cell_size * x_axis + (j - n_cells // 2) * cell_size * y_axis

    return grid


def construct_orthogonal_vec_torch(v):
    # normal . v = 0
    # => normal = [1/v[0], 1/v[1], -1/v[2]]
    v_np = v.numpy()
    axis = [0,0,1] if abs(v_np[1]) < 1e-3 and abs(v_np[2]) < 1e-3 else [1,0,0]
    q0 = pyq.Quaternion(axis=axis, angle=90.0) # Rotate 0 about x=y=z
    v_np /= np.linalg.norm(v_np)
    v_rotated = np.array(q0.rotate(v_np))
    x_axis = np.cross(v_rotated, v_np)
    x_axis /= np.linalg.norm(x_axis)
    return torch.from_numpy(x_axis)


def project_to_plane(center: torch.Tensor, x_axis: torch.Tensor, y_axis: torch.Tensor, x: torch.Tensor):
    x_centered = x - center
    x_proj = center + x_axis * torch.dot(x_axis, x_centered) + y_axis * torch.dot(y_axis, x_centered)
    return x_proj