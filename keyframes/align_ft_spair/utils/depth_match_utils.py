
from typing import List
import torch
import numpy as np
from keyframes.align_ft_spair.utils import ft_align_utils, geom_utils

# TODO what is the difference to construct_plane in geom_utils?
def construct_plane(
    img_embd: torch.Tensor, 
    img_xyz: torch.Tensor,
    kpt_features_avg_train: torch.Tensor,
    stable_indices: List[int],
    oriented_triangles: List[List[int]],
    # stable_threshold = 0.00048,
    stable_threshold_rate = 0.8,
    log_level = 0,
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
    if log_level > 0:
        print("plane_kpt_indices", plane_kpt_indices)
    plane_kpt_indices_set = set(plane_kpt_indices.tolist())
    n_kpts = len(plane_kpt_indices)
    
    plane_kpt_xyz = []
    plane_kpt_idx_to_xyz = {}
    for kpt_index in plane_kpt_indices:
        plane_kpt_xyz.append(geom_utils.extract_point_v2(img_xyz, kpt_attn[kpt_index]))
        plane_kpt_idx_to_xyz[kpt_index] = plane_kpt_xyz[-1]

    # plane center = average of plane_kpt_xyz
    plane_origin = torch.zeros(3).double()
    if len(plane_kpt_indices) == 0 and log_level > 0:
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
            R, t, det = geom_utils.construct_local_coord_system(
                plane_kpt_idx_to_xyz[i1].numpy(),
                plane_kpt_idx_to_xyz[i2].numpy(),
                plane_kpt_idx_to_xyz[i3].numpy()
            )
            if abs(det) > 0.001:
                trianlge_dirs.append(torch.from_numpy(R[2,:]))
    plane_dir = torch.tensor([0,0,1.0]).double()
    if len(trianlge_dirs) == 0 and log_level > 0:
        print("Warning:", "triangles have low determinant or no triangle could be constructed")
        print("plane kpt indices", plane_kpt_indices)
    else:
        plane_dir = torch.mean(torch.stack(trianlge_dirs), dim=0)
    return plane_origin, plane_dir, plane_kpt_xyz, kpt_attn






# def match_kpts(
#     query_embd_coords: torch.Tensor, 
#     img_embd_src: torch.Tensor, 
#     img_embd_tgt: torch.Tensor, 
#     xyz_src: torch.Tensor, 
#     xyz_tgt: torch.Tensor, 
#     kpt_features_avg_train: torch.Tensor
# ):
#     """
#     Args:
#         embd_coords_src: (n_q, 2) 
#         img_embd_src: (C, h, w)
#         img_embd_tgt: (C, h, w)
#         xyz_src: (h*w, 3)
#         xyz_tgt: (h*w, 3)
#         kpt_features_avg_train:  (n_kpts, C)
#     Out:
#         attn: (n_q, h, w)
#     """
#     _, h, w = img_embd_src.shape

#     # compute attention between kpt_features_avg_train and imgs
#     source_kpt_attn = ft_align_utils.compute_attn(kpt_features_avg_train, img_embd_src[None,:,:,:])  # (n_kpts, h, w)
#     target_kpt_attn = ft_align_utils.compute_attn(kpt_features_avg_train, img_embd_tgt[None,:,:,:])  # (n_kpts, h, w)
#     query_embds = img_embd_src[:, query_embd_coords[:,1], query_embd_coords[:,0]]  # (C, n_q)
#     attn = ft_align_utils.compute_attn(query_embds.T, img_embd_tgt[None,:,:,:])  # (n_q, h, w)

#     # select keypoints with maximum confidence across imgs
#     source_kpt_attn_max = torch.max(torch.max(source_kpt_attn, dim=-1).values, dim=-1).values  # (n_kpts,)
#     target_kpt_attn_max = torch.max(torch.max(target_kpt_attn, dim=-1).values, dim=-1).values  # (n_kpts,)
#     kpt_attn_max = torch.minimum(source_kpt_attn_max, target_kpt_attn_max)
#     best_kpts_indices = torch.argsort(kpt_attn_max, descending=True)
#     best_kpts_indices_selected = best_kpts_indices[:4]
#     # print("using the following keypoints:", best_kpts_indices_selected)

#     # extract xyz of keypoints for both imgs
#     best_kpts_xyz_source = []
#     best_kpts_xyz_target = []
#     for kpt_index in best_kpts_indices_selected:
#         best_kpts_xyz_source.append(extract_point(xyz_src, source_kpt_attn[kpt_index]).numpy())
#         best_kpts_xyz_target.append(extract_point(xyz_tgt, target_kpt_attn[kpt_index]).numpy())

#     # construct triangles from keypoints and keep those with non zero determinant
#     n = len(best_kpts_indices_selected)
#     T_all_src, T_all_tgt = [], []
#     for i in range(n):
#         for j in range(i+1, n):
#             for l in range(j+1, n):
#                 R_src, t_src, det_src = construct_local_coord_system(
#                     best_kpts_xyz_source[i],
#                     best_kpts_xyz_source[j],
#                     best_kpts_xyz_source[l],
#                 )
#                 R_tgt, t_tgt, det_tgt = construct_local_coord_system(
#                     best_kpts_xyz_target[i],
#                     best_kpts_xyz_target[j],
#                     best_kpts_xyz_target[l],
#                 )
#                 if min(abs(det_src), abs(det_tgt)) > 0.01:
#                     T_all_src.append(np.concatenate([R_src, -R_src @ t_src[:,None]], axis=-1))
#                     T_all_tgt.append(np.concatenate([R_tgt, -R_tgt @ t_tgt[:,None]], axis=-1))
#     if len(T_all_src) == 0 or len(T_all_tgt) == 0:
#         return attn
#     T_all_src = torch.from_numpy(np.stack(T_all_src))  # (n_triangles, 3, 4)
#     T_all_tgt = torch.from_numpy(np.stack(T_all_tgt))
#     n_triangles = T_all_src.shape[0]

#     # extract xyz for query points
#     query_xyz = xyz_src.reshape((h, w, 3))[query_embd_coords[:,1], query_embd_coords[:,0], :]  # (n_q, 3)

#     # extend coords with 1 (homogeneous coords)
#     query_xyz = torch.cat([query_xyz, torch.ones((query_xyz.shape[0], 1))], dim=-1)
#     xyz_tgt = torch.cat([xyz_tgt, torch.ones((xyz_tgt.shape[0], 1))], dim=-1)

#     # at ground truth target signs should be equal
#     local_coords_src = T_all_src @ query_xyz.T[None,:,:]  # (n_triangles, 3, n_q)
#     local_coords_tgt = T_all_tgt @ xyz_tgt.T[None,:,:]  # (n_triangles, 3, h*w)
    
#     z_coords_q = local_coords_src[:,2,:]  # (n_triangles, n_q)
#     z_coords_tgt = local_coords_tgt[:,2,:]  # (n_triangles, h*w)

#     # only consider sign for weighting (due to poor quality of point cloud)
#     z_sign_q = torch.sign(z_coords_q)
#     z_sign_tgt = torch.sign(z_coords_tgt)
#     # z_sign_q = z_coords_q / torch.max(torch.abs(z_coords_q), dim=0, keepdim=True).values
#     # z_sign_tgt = z_coords_tgt / torch.max(torch.abs(z_coords_tgt), dim=0, keepdim=True).values
    
#     # give coords far away from plane more weight
#     # z_weight_q = F.softmax(torch.abs(z_coords_q), dim=0)
#     # z_weight_tgt = F.softmax(torch.abs(z_coords_tgt), dim=0)

#     # ignore points close to plane
#     z_sign_q[torch.abs(z_coords_q) < 0.001] = 0
#     z_sign_tgt[torch.abs(z_coords_tgt) < 0.001] = 0

#     # compute weighting
    
#     sign_weight_tgt = (z_sign_q).T @ (z_sign_tgt)  # (n_q, h*w)
#     # sign_weight_tgt /= (torch.max(torch.abs(sign_weight_tgt), dim=-1, keepdim=True).values + 1e-6)
#     sign_weight_tgt = F.softmax(sign_weight_tgt, dim=-1)
#     attn = (attn + sign_weight_tgt.reshape((-1, h, w)))
#     # sign_weight_tgt /= torch.sum(sign_weight_tgt, dim=-1, keepdim=True)
#     # attn += 0.1*sign_weight_tgt.reshape((-1, h, w))
#     # attn *= sign_weight_tgt.reshape((-1, h, w))
#     return attn