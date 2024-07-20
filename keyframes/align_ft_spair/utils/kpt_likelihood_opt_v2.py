from typing import Optional, List
import torch
import numpy as np
from keyframes.align_ft_spair.utils import ft_align_utils, spair_utils, geom_utils, kpt_likelihood_utils
from utils import torch_utils
from tqdm import tqdm
from sklearn.cluster import KMeans



def my_energy(
    # kpt_xyz_fixed: torch.Tensor,
    # kpt_xyz_moving: torch.Tensor,
    kpt_xyz: torch.Tensor,
    # kpt_xyz_fixed: torch.Tensor,
    # kpt_fixed_mask: torch.Tensor,
    # target_kpt_index: int,
    target_kpt_indices: List[int],
    target_xyz: torch.Tensor,
    model_energy_weight: float,

    # pre-computed values that define model energy
    kpt_ratios_ref: torch.Tensor,
    kpt_angles_ref: torch.Tensor,
):
    # kpt_xyz = torch.empty((kpt_fixed_mask.shape[0], 3), dtype=kpt_xyz_fixed.dtype)
    # kpt_xyz[kpt_fixed_mask] = kpt_xyz_fixed
    # kpt_xyz[~kpt_fixed_mask] = kpt_xyz_moving

    # 1) compute alignment energy
    alignment_energy = torch.sum((kpt_xyz[target_kpt_indices,:] - target_xyz)**2)

    # 2) contraints
    # constraint_energy = 10000000 * torch.sum((kpt_xyz[kpt_fixed_mask,:] - kpt_xyz_fixed)**2)

    # 2) compute model energy
    kpt_angles_cur, kpt_ratios_cur = kpt_likelihood_utils.compute_angles_and_ratios(kpt_xyz)
    model_angle_energy = torch.sum((kpt_angles_cur - kpt_angles_ref)**2)
    model_ratio_energy = torch.sum((kpt_ratios_cur - kpt_ratios_ref)**2)
    model_energy = model_angle_energy + model_ratio_energy # model_angle_energy # + 

    # 4) total energy
    total_energy = alignment_energy + model_energy_weight * model_energy  # alignment_energy  + constraint_energy
    
    return total_energy


# TODO function needs to account for constraints (fixed keypoints)
# TODO only optimize one keypoint at a time
def update_kpt_xyz(
    kpt_xyz: torch.Tensor,
    # kpt_fixed_mask: torch.Tensor, 
    target_kpt_indices: List[int],
    target_xyz: torch.Tensor,

    # pre-computed values that define model energy
    kpt_ratios_ref: torch.Tensor,
    kpt_angles_ref: torch.Tensor,

    # params
    lr: float = 0.1,
    n_iter: int = 100,
    model_energy_weight: float = 0.01,
):
    """ update kpt_xyz such that kpt_xyz[target_kpt_index] moves towards target_xyz while minimizing energy
    Args:
        - kpt_xyz: (K, 3) 
        - kpt_fixed_mask: (K,)
        - target_kpt_indices: list of length T
        - target_xyz: (T,3)
        - kpt_ratios_ref: (K**3, 1)
        - kpt_angles_ref: (K**3, 1)
        - lr: learning rate
        - n_iter: number of iterations
        - model_energy_weight: weight of model energy
    """
    # kpt_xyz_fixed = kpt_xyz.clone().detach()[kpt_fixed_mask,:]
    # kpt_xyz_moving = kpt_xyz.clone().detach()[~kpt_fixed_mask,:]
    # kpt_xyz_moving = kpt_xyz_moving.clone().detach().requires_grad_(True)
    # kpt_xyz_fixed = kpt_xyz.clone().detach()[kpt_fixed_mask,:]
    kpt_xyz_moving = kpt_xyz.clone().detach().requires_grad_(True)
    # optimizer = torch.optim.LBFGS([kpt_xyz], lr=lr)
    optimizer = torch.optim.Adam([kpt_xyz_moving], lr=lr)

    for i in tqdm(range(n_iter)):
        optimizer.zero_grad()
        total_energy = my_energy(
            # kpt_xyz_fixed=kpt_xyz_fixed,
            # kpt_xyz_moving=kpt_xyz_moving,
            kpt_xyz=kpt_xyz_moving,
            # kpt_xyz_fixed=kpt_xyz_fixed,
            # kpt_fixed_mask=kpt_fixed_mask,
            target_kpt_indices=target_kpt_indices,
            target_xyz=target_xyz,
            kpt_ratios_ref=kpt_ratios_ref,
            kpt_angles_ref=kpt_angles_ref,
            model_energy_weight=model_energy_weight,
        )
        total_energy.backward()
        optimizer.step()

    kpt_xyz_res = kpt_xyz_moving.clone().detach()
    # kpt_xyz_res[~kpt_fixed_mask,:] = kpt_xyz_moving.detach()
    return kpt_xyz_res


def eval_alignment_energy_for_candidates(
    kpt_xyz: torch.Tensor,
    candidates_xyz: torch.Tensor,
    kpt_likelihood_bias: torch.Tensor
):
    """
    Args:
        - kpt_xyz: (K, 3)
        - candidates_xyz: (N, 3)
        - kpt_likelihood_bias: (K, N)
    """
    k, n = kpt_likelihood_bias.shape
    # TODO add ambiguity
    # energy is squared distance to closest candidate weighted by likelihood
    # compute squared distance
    dists = torch.cdist(kpt_xyz, candidates_xyz, p=2)  # (K, N)
    # compute energy
    min_dist_arg = torch.argmin(dists / (kpt_likelihood_bias + 1e-6), dim=1)  # (K,)

    kpt_indices = torch.arange(k)
    energy = torch.sum(dists[kpt_indices, min_dist_arg] * kpt_likelihood_bias[kpt_indices, min_dist_arg])
    
    return energy


def assign_keypoints_from_fitted_model(
    kpt_xyz: torch.Tensor,
    candidates_xyz: torch.Tensor,
    kpt_likelihood_bias: torch.Tensor,
    candidate_threshold: float = 0.3,
):
    """ assign keypoint indices to candidates
    """
    point_pairs = torch.tensor([
        [4,5],
        [6,7],
        [8,9],
        [10,11],
        [12,13],
        [14,15],
        [16,17],
        [18,19],
        [20,21],
    ], dtype=torch.int)

    point_pair_edges = kpt_xyz[point_pairs[:,1],:] - kpt_xyz[point_pairs[:,0],:]
    point_pair_edge_norms = torch.norm(point_pair_edges, dim=1)

    # project candidates on point_pair_edges
    projected_candidates = (point_pair_edges / point_pair_edge_norms[:,None])[:,None,:] @ (candidates_xyz[None,:,:] - kpt_xyz[point_pairs[:,0],:][:,None,:]).swapaxes(1,2)  # (9, N)
    projected_candidates = projected_candidates.squeeze(1)  # (9,N)
    # some processing (ignore candidates that are likely not candidates)
    # TODO these candidates should not be candidates to begin with
    # candidate_with_max_likelihood = torch.max(kpt_likelihood_bias, dim=1).values  # (K,)
    # candidate_mask = kpt_likelihood_bias > (candidate_with_max_likelihood[:,None] * candidate_threshold)

    candidate_indices = torch.arange(candidates_xyz.shape[0])
    # candidate_kpt_labels = -torch.ones((candidates_xyz.shape[0],), dtype=torch.int)

    pred_kpt_labels = torch.argmax(kpt_likelihood_bias, dim=0)

    candidate_kpt_labels = pred_kpt_labels.int()

    for i in range(point_pairs.shape[0]):
        i1,i2 = point_pairs[i]

        candidate_mask_i1 = pred_kpt_labels == i1
        candidate_mask_i2 = pred_kpt_labels == i2
        projected_candidates_i1 = projected_candidates[i,:][candidate_mask_i1]
        projected_candidates_i2 = projected_candidates[i,:][candidate_mask_i2]
        candidate_indices_i1 = candidate_indices[candidate_mask_i1]
        candidate_indices_i2 = candidate_indices[candidate_mask_i2]
        cat_projected_candidates = torch.cat([projected_candidates_i1, projected_candidates_i2], dim=0)

        # projected_candidates_i1 = projected_candidates[i,:][candidate_mask[i1,:]]
        # projected_candidates_i2 = projected_candidates[i,:][candidate_mask[i2,:]]
        # candidate_indices_i1 = candidate_indices[candidate_mask[i1,:]]
        # candidate_indices_i2 = candidate_indices[candidate_mask[i2,:]]
        # cat_projected_candidates = torch.cat([projected_candidates_i1, projected_candidates_i2], dim=0)



        cat_candidate_indices = torch.cat([candidate_indices_i1, candidate_indices_i2], dim=0)
        cat_candidates_xyz = candidates_xyz[cat_candidate_indices,:]
        if cat_projected_candidates.shape[0] > 0:
            min_cluster_dist = point_pair_edges[i].norm() * 0.1

            # cluster candidates
            # data = cat_projected_candidates.view(-1, 1).numpy()
            data = cat_candidates_xyz.numpy()

            if data.shape[0] >= 2:
                # Number of clusters
                k = 2

                # Perform K-means clustering
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(data)

                # Get the centroids and labels
                centroids = torch.from_numpy(kmeans.cluster_centers_)
                cluster_indices = torch.from_numpy(kmeans.labels_).int()

            else:
                centroids = torch.cat([torch.from_numpy(data),torch.from_numpy(data)], dim=0)

            centroids_projected = centroids @ point_pair_edges[i] / point_pair_edge_norms[i]
            centroids_dist = torch.norm(centroids[1] - centroids[0])

            # if clusters are too close, assign all candidates to only one keypoint
            if centroids_dist < min_cluster_dist:
                # assign cluster to keypoint with closest distance
                candidate_mean = torch.mean(cat_projected_candidates, dim=0)
                dist_to_i1 = torch.abs(candidate_mean)
                dist_to_i2 = torch.abs(candidate_mean - point_pair_edge_norms[i])
                if dist_to_i1 < dist_to_i2:
                    candidate_kpt_labels[cat_candidate_indices] = i1
                else:
                    candidate_kpt_labels[cat_candidate_indices] = i2
            else:
                # assign clusters to keypoint with closest distance
                if centroids_projected[1] - centroids_projected[0] > 0:
                    candidate_kpt_labels[cat_candidate_indices[cluster_indices == 0]] = i1
                    candidate_kpt_labels[cat_candidate_indices[cluster_indices == 1]] = i2
                else:
                    candidate_kpt_labels[cat_candidate_indices[cluster_indices == 1]] = i1
                    candidate_kpt_labels[cat_candidate_indices[cluster_indices == 0]] = i2
    
    return candidate_kpt_labels


def get_next_kpt_index(
    kpt_likelihood_bias: torch.Tensor,
    # candidates_xyz: torch.Tensor,
    # kpt_xyz_cur: torch.Tensor,
    # kpt_xyz_ref: torch.Tensor,
    kpt_chosen_cur: torch.Tensor,
    candidate_chosen_cur: torch.Tensor,

    preferred_keypoint_order: torch.Tensor,
    threshold: float = 0.8,

    # params
    # kpt_is_similar_threshold=0.4,
    # kpt_is_similar_threshold_energy=0.4,
    # kpt_min_likelihood=0.6,
):
    """
    Args:
        - kpt_likelihood_bias: (K, N)
        - kpt_chosen_cur: (K,)
        - candidate_chosen_cur: (N,)
        - preferred_keypoint_order: (K,) preferred_keypoint_order[i] is the position at which keypoint i should be assigned
    """
    k, n = kpt_likelihood_bias.shape
    not_chosen = torch.logical_not(kpt_chosen_cur)
    not_chosen_candidates = torch.logical_not(candidate_chosen_cur)
    kpt_indices = torch.arange(k)
    candidate_indices = torch.arange(n)
    kpt_likelihood_bias_of_not_chosen = kpt_likelihood_bias[not_chosen,:][:, not_chosen_candidates]
    kpt_indices_not_chosen = kpt_indices[not_chosen]
    candidate_indices_not_chosen = candidate_indices[not_chosen_candidates]
    preferred_keypoint_order_not_chosen = preferred_keypoint_order[not_chosen]


    # if kpt_likelihood_bias_of_not_chosen.numel() == 0:
    #     return kpt_xyz_cur.clone(), kpt_chosen_cur.clone(), candidate_chosen_cur.clone()
    
    # 2) get similar keypoints to kpt_index at candidate location
    # TODO there should be also an ambiguity term across candidate locations
    # => We could also do this in post-processing, refine assignment to other candidate location after all keypoints have been assigned
    kpt_ambiguity_likelihood_not_chosen = kpt_likelihood_bias_of_not_chosen / (torch.sum(kpt_likelihood_bias_of_not_chosen, dim=0, keepdim=True) + 1e-6)
    # similar_kpt_indices_mask = torch.logical_and(
    #     kpt_ambiguity_likelihood_not_chosen[kpt_index_in_not_chosen] > 1/(1+kpt_is_similar_threshold),
    #     candidate_indices_not_chosen != kpt_index
    # )
    # similar_kpt_indices = candidate_indices_not_chosen[similar_kpt_indices_mask]

    total_likelihood_not_chosen = kpt_ambiguity_likelihood_not_chosen  # * kpt_likelihood_bias_of_not_chosen

    
    # TODO we should first compute kpt_ambiguity likelihood (combine likelihood and energy with max), 
    #   then multiply with likelihood_bias to retrieve max likelihood
    # 1) get candidate xyz with maximum likelihood
    kpt_total_likelihood_argmax_not_chosen = torch.argmax(total_likelihood_not_chosen, dim=-1) # (K,)
    kpt_max_likelihoods_not_chosen = total_likelihood_not_chosen[torch.arange(total_likelihood_not_chosen.shape[0]), kpt_total_likelihood_argmax_not_chosen] # (K,)
    sort_res_not_chosen = torch.sort(kpt_max_likelihoods_not_chosen, descending=True)
    considered_keypoints_mask = sort_res_not_chosen.values > threshold * sort_res_not_chosen.values[0]
    considered_keypoint_indices_in_not_chosen = sort_res_not_chosen.indices[considered_keypoints_mask]
    # from considered keypoints, get the one that is the first in the preferred_keypoint_order
    kpt_index_in_not_chosen = considered_keypoint_indices_in_not_chosen[
        torch.argmin(preferred_keypoint_order_not_chosen[considered_keypoint_indices_in_not_chosen])
    ]
    # kpt_index_in_not_chosen = sort_res_not_chosen.indices[0]
    kpt_index = kpt_indices_not_chosen[kpt_index_in_not_chosen]
    candidate_index_in_not_chosen = kpt_total_likelihood_argmax_not_chosen[kpt_index_in_not_chosen]
    candidate_index = candidate_indices_not_chosen[candidate_index_in_not_chosen]

    return kpt_index, candidate_index



    # # 1.2) if kpt_likelihood is below threshold, return
    # if kpt_max_likelihoods_not_chosen[0] < kpt_min_likelihood:
    #     return kpt_xyz_cur.clone(), kpt_chosen_cur.clone(), candidate_chosen_cur.clone()

    

    # # 3) evaluate energy at candidate location which might disambiguate the similar keypoints
    # if len(similar_kpt_indices) > 0:
    #     model_energy_not_chosen_at_candidate = kpt_likelihood_utils.eval_energy_for_query_xyz(
    #         kpt_xyz_cur, kpt_chosen_cur, candidates_xyz[candidate_index,:], kpt_xyz_ref
    #     ) # (#not_chosen, 1)
    #     model_energy_likelihood_not_chosen = torch.exp(-model_energy_not_chosen_at_candidate).float()
    #     kpt_energy_ambiguity_not_chosen = model_energy_likelihood_not_chosen / (torch.sum(model_energy_likelihood_not_chosen, dim=0, keepdim=True) + 1e-6)
    #     similar_kpt_indices_mask = torch.logical_and(
    #         kpt_energy_ambiguity_not_chosen[kpt_index_in_not_chosen] > 1/(1+kpt_is_similar_threshold_energy),
    #         similar_kpt_indices_mask
    #     )
    #     similar_kpt_indices = candidate_indices_not_chosen[similar_kpt_indices_mask]


def extract_peaks(
    img_xyz_orig: torch.Tensor,
    img_kpt_label_likelihood: torch.Tensor,
    max_num_peaks=3,
    min_distance=10,
    min_peak_value=0.01,
):
    """
    NOTE this is an old version, look at peak_extraction_utils for newest version
    Args:
        - img_xyz_orig: (H_orig, W_orig, 3)
        - img_kpt_label_likelihood: (K, H, W)
    """

    embd_size = img_kpt_label_likelihood.shape[1]

    # 1) extract peaks (candidates)
    peak_xyz_coords_all, peak_xy_coords_all, peak_values_all = ft_align_utils.extract_peaks_xyz_for_all_kpts(
        kpt_attn=img_kpt_label_likelihood,
        img_xyz_orig=img_xyz_orig,
        max_num_peaks=max_num_peaks,
        min_distance=min_distance,
        min_peak_value=min_peak_value,
    )

    # concatenate all peaks
    peak_xyz_coords_all_concat = torch.cat(peak_xyz_coords_all, dim=0)
    kpt_indices = torch.arange(len(peak_xyz_coords_all))
    peak_concat_indices = [[int(i)]*len(peak_xyz_coords_all[i]) for i in kpt_indices if len(peak_xyz_coords_all[i]) > 0]
    peak_concat_indices = [item for sublist in peak_concat_indices for item in sublist]

    # transform peak img coords to embd coords
    peak_embd_coords_xy = []
    for i in range(len(peak_xy_coords_all)):
        peaks_xy = peak_xy_coords_all[i]
        for p in range(peaks_xy.shape[0]):
            x_new, y_new = spair_utils.transform_image_coords(
                x_orig=peaks_xy[p, 0],
                y_orig=peaks_xy[p,1], 
                img_orig_width=img_xyz_orig.shape[1], 
                img_orig_height=img_xyz_orig.shape[0], 
                img_new_size=embd_size, 
                pad=True
            )
            peak_embd_coords_xy.append([x_new, y_new])
    peak_embd_coords_xy = torch.tensor(peak_embd_coords_xy)


    # create peak_kpt_likelihood and peak_xyz tensor
    if peak_embd_coords_xy.shape[0] > 0:
        peak_kpt_likelihood = img_kpt_label_likelihood[:,peak_embd_coords_xy[:,1], peak_embd_coords_xy[:,0]]
        peak_xyz = peak_xyz_coords_all_concat
    else:
        print("No peaks found")
        peak_kpt_likelihood = torch.empty((img_kpt_label_likelihood.shape[0], 0))
        peak_xyz = torch.empty((0, 3))

    return peak_xyz, peak_kpt_likelihood, peak_concat_indices, peak_embd_coords_xy


def fit_model(
    # reference geometric relationships of airplane
    kpt_xyz_ref: torch.Tensor,
    kpt_angles_ref: torch.Tensor,
    kpt_ratios_ref: torch.Tensor,
    preferred_keypoint_order: torch.Tensor,

    # candidates (peaks)
    peak_xyz: torch.Tensor,
    peak_kpt_likelihood: torch.Tensor,

    # params
    threshold = 0.8,
    n_steps = 5,
):
    
    # ===================
    # THE ALGORITHM
    # ===================

    # 0) initialization
    kpt_chosen_cur = torch.zeros((peak_kpt_likelihood.shape[0],), dtype=torch.bool)
    candidate_chosen_cur = torch.zeros((peak_kpt_likelihood.shape[1],), dtype=torch.bool)
    k,n = peak_kpt_likelihood.shape
    kpt_indices = torch.arange(k)

    # 1) choose two keypoints 
    # (assumption is that in each image there are at least two stable keypoint indices)
    for i in range(2):
        kpt_index, candidate_index = get_next_kpt_index(
            kpt_likelihood_bias=peak_kpt_likelihood,
            kpt_chosen_cur=kpt_chosen_cur,
            candidate_chosen_cur=candidate_chosen_cur,
            preferred_keypoint_order=preferred_keypoint_order,
            threshold=threshold
        )
        kpt_chosen_cur[kpt_index] = True
        candidate_chosen_cur[candidate_index] = True
        print(kpt_index)

    # 2) initialize kpt_xyz_opt
    target_indices = kpt_indices[kpt_chosen_cur].numpy().tolist()
    target_xyz = peak_xyz[candidate_chosen_cur,:]
    scale = torch.norm(target_xyz[1,:] - target_xyz[0,:]) / torch.norm(kpt_xyz_ref[target_indices[1],:] - kpt_xyz_ref[target_indices[0],:])
    kpt_xyz_opt = kpt_xyz_ref.clone() * scale

    # 3) optimization procedure
    for i in range(n_steps):
        # pick next potential keypoints
        kpt_index, candidate_index = get_next_kpt_index(
            kpt_likelihood_bias=peak_kpt_likelihood,
            kpt_chosen_cur=kpt_chosen_cur,
            candidate_chosen_cur=candidate_chosen_cur,
            preferred_keypoint_order=preferred_keypoint_order,
            threshold=threshold
        )
        if kpt_index in [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]:
            kpt_index_1 = kpt_index
            if kpt_index_1 % 2 == 0:
                kpt_index_2 = kpt_index + 1
            else:
                kpt_index_2 = kpt_index - 1
            if i > 10:
                # project
                candidate_kpt_indices = assign_keypoints_from_fitted_model(
                    kpt_xyz=kpt_xyz_opt,
                    candidates_xyz=peak_xyz,
                    kpt_likelihood_bias=peak_kpt_likelihood,
                    candidate_threshold=0.2
                )
                kpt_index = candidate_kpt_indices[candidate_index]
            else:
                # check for both keypoint indices
                candidate_chosen_cur[candidate_index] = True
                energies = []
                for kpt_index in [kpt_index_1, kpt_index_2]:
                    print("intermediate", kpt_index)
                    kpt_chosen_tmp = kpt_chosen_cur.clone()
                    kpt_chosen_tmp[kpt_index] = True
                    target_indices = kpt_indices[kpt_chosen_tmp].numpy().tolist()
                    target_xyz = peak_xyz[candidate_chosen_cur,:]
                    kpt_xyz_opt = update_kpt_xyz(
                        kpt_xyz=kpt_xyz_opt,
                        # kpt_fixed_mask=kpt_fixed_mask, 
                        # target_kpt_index=int(target_kpt_index),
                        target_kpt_indices=target_indices,
                        target_xyz=target_xyz,

                        # pre-computed values that define model energy
                        kpt_angles_ref=kpt_angles_ref,
                        kpt_ratios_ref=kpt_ratios_ref,

                        lr=0.001, # if we use only quadratic alignment energy, lr=1 works
                        n_iter=1000,
                        model_energy_weight=0.0001,
                    )
                    energy = eval_alignment_energy_for_candidates(
                        kpt_xyz = kpt_xyz_opt,
                        candidates_xyz=peak_xyz,
                        kpt_likelihood_bias=peak_kpt_likelihood
                    )
                    energies.append(energy)
                kpt_index = kpt_index_1 if energies[0] < energies[1] else kpt_index_2
        print(kpt_index)
        kpt_chosen_cur[kpt_index] = True
        candidate_chosen_cur[candidate_index] = True

    # optimize again
    target_indices = kpt_indices[kpt_chosen_cur].numpy().tolist()
    target_xyz = peak_xyz[candidate_chosen_cur,:]
    kpt_xyz_opt = update_kpt_xyz(
        kpt_xyz=kpt_xyz_opt,
        # kpt_fixed_mask=kpt_fixed_mask, 
        # target_kpt_index=int(target_kpt_index),
        target_kpt_indices=target_indices,
        target_xyz=target_xyz,

        # pre-computed values that define model energy
        kpt_angles_ref=kpt_angles_ref,
        kpt_ratios_ref=kpt_ratios_ref,

        lr=0.001, # if we use only quadratic alignment energy, lr=1 works
        n_iter=10000,
        model_energy_weight=0.0001,
    )

    return kpt_xyz_opt


def eval_model(
    img_kpt_label_likelihood: torch.Tensor,
    img_xyz_orig: torch.Tensor,
    kpt_img_coords: torch.Tensor,
    kpt_xyz_opt: torch.Tensor,
    thresh_rate: float = 0.9,
    window_size: int = 10,
):
    """ Assign labels to kpt_img_coords based on kpt_xyz_opt and compare with ground truth labels
    Args:
        - img_kpt_likelihood: (K, H, W)
        - img_xyz_orig: (H_orig, W_orig, 3)
        - kpt_img_coords: (K, 3)
        - kpt_xyz_opt: (K, 3)
    """

    # queries are the ground truth keypoints
    query_xy = kpt_img_coords[:,0:2]
    query_xyz = ft_align_utils.kpt_img_coords_to_xyz_with_correction(
        kpt_img_coords_xy=query_xy,
        kpt_labels=kpt_img_coords[:,2],
        img_xyz_orig=img_xyz_orig,
        img_kpt_label_likelihood=img_kpt_label_likelihood,
        thresh_rate=thresh_rate,
        window_size=window_size,
    )
    gt_query_labels = kpt_img_coords[:,2]
    embd_size = img_kpt_label_likelihood.shape[1]
    n = query_xy.shape[0] # n keypoint (candidates)

    # transform query img coords to embd coords
    # TODO should be done in a separate function => and parallelized
    query_embd_coords_xy = []
    for i in range(n):
        x_new, y_new = spair_utils.transform_image_coords(
            x_orig=int(query_xy[i, 0]),
            y_orig=int(query_xy[i,1]),
            img_orig_width=img_xyz_orig.shape[1],
            img_orig_height=img_xyz_orig.shape[0],
            img_new_size=embd_size,
            pad=True
        )
        query_embd_coords_xy.append([x_new, y_new])
    query_embd_coords_xy = torch.tensor(query_embd_coords_xy)

    # extract kpt label likelihood at kpt locations
    kpt_label_likelihood_at_query_locations = img_kpt_label_likelihood[:,query_embd_coords_xy[:,1], query_embd_coords_xy[:,0]]

    pred_query_kpt_labels = assign_keypoints_from_fitted_model(
        kpt_xyz=kpt_xyz_opt,
        candidates_xyz=query_xyz,
        kpt_likelihood_bias=kpt_label_likelihood_at_query_locations,
        candidate_threshold=0.2
    )

    # compute accuracy
    accuracy = torch.sum(pred_query_kpt_labels == gt_query_labels) / n
    
    return accuracy, pred_query_kpt_labels, query_xyz, kpt_label_likelihood_at_query_locations