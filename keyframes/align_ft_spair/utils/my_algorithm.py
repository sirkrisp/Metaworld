from typing import Callable, Optional, List
import torch
import numpy as np
from keyframes.align_ft_spair.utils import ft_align_utils, spair_utils, geom_utils, kpt_likelihood_utils
from utils import torch_utils
from tqdm import tqdm
from sklearn.cluster import KMeans


def get_candidates():
    """ Get points in image that have likelihood higher than threshold for at least one keypoint
        label and are inside the target object. (also extract peaks? => probably)
    """
    pass


# NOTE same as get_next_keypoint in kpt_likelihood_opt_v2
def select_keypoint(
    kpt_label_likelihood: torch.Tensor,
    kpts_chosen: torch.Tensor,
    candidates_chosen: torch.Tensor,
    preferred_kpt_order: torch.Tensor,
    threshold: float = 0.8,
):
    """
    Args:
        - kpt_label_likelihood: (K, N) Keypoint label likelihood for each candidate
        - kpts_chosen: (K,) if kpts_chosen[i] is True, keypoint i will be ignored
        - candidates_chosen: (N,) if candidates_chosen[i] is True, candidate i will be ignored
        - preferred_keypoint_order: (K,) preferred_keypoint_order[i] is the position at which keypoint i should be assigned
    """

    # setup tensors
    k, n = kpt_label_likelihood.shape
    kpts_not_chosen = torch.logical_not(kpts_chosen)
    candidates_not_chosen = torch.logical_not(candidates_chosen)
    kpt_indices = torch.arange(k)
    # candidate_indices = torch.arange(n)
    kpt_label_likelihood_not_chosen = kpt_label_likelihood[kpts_not_chosen,:][:, candidates_not_chosen]
    kpt_indices_not_chosen = kpt_indices[kpts_not_chosen]
    # candidate_indices_not_chosen = candidate_indices[candidates_not_chosen]
    preferred_keypoint_order_not_chosen = preferred_kpt_order[kpts_not_chosen]

    # compute certainty of keypoint label for each candidate
    kpt_ambiguity_likelihood_not_chosen = kpt_label_likelihood_not_chosen / (torch.sum(kpt_label_likelihood_not_chosen, dim=0, keepdim=True) + 1e-6)
    total_likelihood_not_chosen = kpt_ambiguity_likelihood_not_chosen * kpt_label_likelihood_not_chosen

    # TODO maybe we should add back energy likelihood
    # basically depends on whether we want hard or soft boundary
    # mix by taking average

    # select a set of keypoint labels for which there exist a candidate with likelihood higher
    # than threshold * max_likelihood
    kpt_total_likelihood_argmax_not_chosen = torch.argmax(total_likelihood_not_chosen, dim=-1) # (K,)
    kpt_max_likelihoods_not_chosen = total_likelihood_not_chosen[torch.arange(total_likelihood_not_chosen.shape[0]), kpt_total_likelihood_argmax_not_chosen] # (K,)
    sort_res_not_chosen = torch.sort(kpt_max_likelihoods_not_chosen, descending=True)
    considered_keypoints_mask = sort_res_not_chosen.values > threshold * sort_res_not_chosen.values[0]
    considered_keypoint_indices_in_not_chosen = sort_res_not_chosen.indices[considered_keypoints_mask]
    
    # from considered keypoints, get the one that is the first in the preferred_keypoint_order
    kpt_index_in_not_chosen = considered_keypoint_indices_in_not_chosen[
        torch.argmin(preferred_keypoint_order_not_chosen[considered_keypoint_indices_in_not_chosen])
    ]
    kpt_index = kpt_indices_not_chosen[kpt_index_in_not_chosen]
    kpt_max_likelihood = kpt_max_likelihoods_not_chosen[kpt_index_in_not_chosen]

    return kpt_index, kpt_max_likelihood


def select_candidates_for_keypoint(
    kpt_index: int,
    kpt_label_likelihood: torch.Tensor,
    candidates_chosen: torch.Tensor,
    threshold: float = 0.8,
):
    """ Select candidates for a keypoint based on the likelihood of the keypoint label
    Args:
        - kpt_index: index of the keypoint to be assigned
        - kpt_label_likelihood: (K, N) Keypoint label likelihood for each candidate
        - kpts_chosen: (K,) if kpts_chosen[i] is True, keypoint i will be ignored
        - candidates_chosen: (N,) if candidates_chosen[i] is True, candidate i will be ignored
    """
    # TODO again here we might should consider energy likelihood
    # TODO maybe we can just select k best candidates
    max_likelihood = torch.max(kpt_label_likelihood[kpt_index, torch.logical_not(candidates_chosen)])
    candidate_mask = kpt_label_likelihood[kpt_index] > threshold * max_likelihood
    candidate_mask = torch.logical_and(candidate_mask, torch.logical_not(candidates_chosen))
    candidate_indices = torch.arange(kpt_label_likelihood.shape[1])[candidate_mask]
    return candidate_indices


def compute_target_alignment_energy(
    kpt_xyz: torch.Tensor,
    target_kpt_indices: List[int],
    target_xyz: torch.Tensor,
):
    """ During optimization, we want that the model keypoints are close to the target keypoints
    """
    # energy is based on squared distance
    target_alignment_energy = torch.sum((kpt_xyz[target_kpt_indices,:] - target_xyz)**2)
    return target_alignment_energy


def compute_candidate_alignment_energy(
    kpt_xyz: torch.Tensor,
    kpt_label_likelihood: torch.Tensor,
    candidates_xyz: torch.Tensor,
):
    """
    energy is squared distance to closest candidate weighted by likelihood
    """

    # compute minimum distances
    k, n = kpt_label_likelihood.shape
    dists = torch.cdist(kpt_xyz, candidates_xyz, p=2)  # (K, N)
    min_dist_arg = torch.argmin(dists / (kpt_label_likelihood + 1e-6), dim=1)  # (K,)

    # compute energy based on minimum distances
    kpt_indices = torch.arange(k)
    candidate_alignment_energy = torch.sum(dists[kpt_indices, min_dist_arg] * kpt_label_likelihood[kpt_indices, min_dist_arg])
    
    return candidate_alignment_energy


def compute_internal_model_energy(
    kpt_xyz: torch.Tensor,
    kpt_cos_angles_mean: torch.Tensor,
    kpt_cos_angles_var: torch.Tensor,
    kpt_ratios_mean: torch.Tensor,
    kpt_ratios_var: torch.Tensor,
    alpha: float = 0.5,
):
    cos_angles, _, ratios = kpt_likelihood_utils.compute_angles_and_ratios_parallel(kpt_xyz[None,:,:])
    cos_angles, ratios = cos_angles.squeeze(0), ratios.squeeze(0)

    # compute log likelihood of angles & ratios
    cos_angles_log_likelihood = torch.sum(-0.5 * (cos_angles - kpt_cos_angles_mean)**2 / kpt_cos_angles_var)
    ratios_log_likelihood = torch.sum(-0.5 * (ratios - kpt_ratios_mean)**2 / kpt_ratios_var)
    
    internal_model_energy = alpha * cos_angles_log_likelihood + (1 - alpha) * ratios_log_likelihood

    return internal_model_energy


def compute_likelihood(x, x_mean, x_var):
    """
    Compute likelihood of x being in range [-infty, -x_hat] and [x_hat, infty],
    where x_hat = |x - mean| and the normal distribution is defined by mean=0 and sd.
    NOTE: at x_hat=0, this value is 1.
    
    Args:
        - x: (D1,D2) D1=#samples, D2=#distributions (or swapped)
        - x_mean: (D1,1) or (1,D2) depending on x
        - x_var: (D1,1) or (1,D2) depending on x
    Returns:
        - likelihood: (D1, D2) tensor
    """
    # Standard deviation is the square root of variance
    x_sd = torch.sqrt(x_var)
    x_sd[torch.isnan(x_sd)] = 1e-13
    x_sd[x_sd < 1e-9] = 1e-13
    
    # Create a normal distribution with mean=0 and the given standard deviation
    normal_dist = torch.distributions.Normal(0, x_sd)
    
    # Compute x_hat = |x - x_mean|
    x_hat = torch.abs(x - x_mean)
    x_hat[torch.isnan(x_hat)] = 1e6
    
    # Compute the cumulative distribution function (CDF) for x_hat
    # Since CDF is symmetric around the mean, use the CDF of x_hat directly
    cdf_x_hat = normal_dist.cdf(x_hat)
    
    # The likelihood is 2 * (1 - CDF(x_hat)) because CDF is symmetric
    likelihood = 2 * (1 - cdf_x_hat)
    
    return likelihood


def compute_candidate_internal_model_energy(
    # kpt_xyz: torch.Tensor,
    candidates_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    target_kpt_labels: torch.Tensor,
    kpt_cos_angles_mean: torch.Tensor,
    kpt_cos_angles_var: torch.Tensor,
    kpt_ratios_mean: torch.Tensor,
    kpt_ratios_var: torch.Tensor,
    signed_volumes_mean: torch.Tensor,
    signed_volumes_var: torch.Tensor,
    compute_signed_volumes_parallel: Callable[[torch.Tensor], torch.Tensor],
):
    """
    Args:
        - candidates_xyz: (C, 3) candidate points for the keypoints
        - target_xyz: (N,)
        - target_kpt_labels: (N),
        - kpt_cos_angles_mean: (K, K, K) mean cos angles for each keypoint
        - kpt_cos_angles_var: (K, K, K) variance of cos angles for each keypoint
        - kpt_ratios_mean: (K, K, K) mean ratios for each keypoint
        - kpt_ratios_var: (K, K, K) variance of ratios for each keypoint
        - signed_volumes_mean: (n_combinations,) mean signed volumes for each keypoint
        - signed_volumes_var: (n_combinations,) variance of signed volumes for each keypoint

    Return:
        - geom_min_log_likelihood: (K, C) minimum log likelihood for each keypoint and candidate
    """
    k = kpt_cos_angles_mean.shape[0]
    c = candidates_xyz.shape[0]
    device = target_xyz.device

    if target_kpt_labels.shape[0] < 2:
        return torch.ones((k, c), device=device)

    # keypoint, compute likelihood at candidate position
    cur_xyz = torch.full((k, 3), fill_value=torch.nan, device=device)
    cur_xyz[target_kpt_labels,:] = target_xyz

    # repeat cur_xyz K*C times to compute angles and ratios
    cur_xyz = cur_xyz[None,None,:,:].repeat(k, c, 1, 1)

    # fill cur_xyz[i,j,i,:] with kpt_candidates_xyz[i,j,:] if i is not in target_indices in parallel
    kpt_indices = torch.arange(k, device=device)
    free_kpt_labels = torch.logical_not(torch.isin(kpt_indices, target_kpt_labels))
    cur_xyz[free_kpt_labels, :, free_kpt_labels, :] = candidates_xyz[None, :, :].repeat(k - len(target_kpt_labels), 1, 1)
    cur_xyz = cur_xyz.reshape((k*c, k, 3))

    # compute angles and ratios
    cos_angles, _, ratios = kpt_likelihood_utils.compute_angles_and_ratios_parallel(cur_xyz)
    signed_volumes = compute_signed_volumes_parallel(cur_xyz)

    # compute log likelihood of angles & ratios
    # TODO we should only consider values that involve keypoint k
    # cos_angles_log_likelihood = -0.5 * (cos_angles - kpt_cos_angles_mean[None,:,:,:])**2 / kpt_cos_angles_var[None,:,:,:]
    # ratios_log_likelihood = -0.5 * (ratios - kpt_ratios_mean[None,:,:,:])**2 / kpt_ratios_var[None,:,:,:]
    # signed_volumes_log_likelihood = -0.5 * (signed_volumes - signed_volumes_mean[None,:])**2 / signed_volumes_var[None,:]
    # cos_angles_likelihood = torch.exp(cos_angles_log_likelihood) / torch.sqrt(2 * torch.pi * kpt_cos_angles_var[None,:,:,:])
    # ratios_likelihood = torch.exp(ratios_log_likelihood) / torch.sqrt(2 * torch.pi * kpt_ratios_var[None,:,:,:])
    # signed_volumes_likelihood = torch.exp(signed_volumes_log_likelihood) / torch.sqrt(2 * torch.pi * signed_volumes_var[None,:])

    # compute likelihood from log likelihood
    cos_angles_likelihood = compute_likelihood(cos_angles, kpt_cos_angles_mean[None,:,:,:], kpt_cos_angles_var[None,:,:,:])
    ratios_likelihood = compute_likelihood(ratios, kpt_ratios_mean[None,:,:,:], kpt_ratios_var[None,:,:,:])
    signed_volumes_likelihood = compute_likelihood(signed_volumes, signed_volumes_mean[None,:], signed_volumes_var[None,:])

    # cos_angles_likelihood[cos_angles_likelihood < 1e-8] = 1
    # ratios_likelihood[ratios_likelihood < 1e-8] = 1
    # signed_volumes_likelihood[signed_volumes_likelihood < 1e-8] = 1
    
    # cos_angles_likelihood_min = torch.min(cos_angles_likelihood, dim=1).values
    # ratios_likelihood_min = torch.min(ratios_likelihood, dim=1).values
    # signed_volumes_likelihood_min = torch.min(signed_volumes_likelihood, dim=1).values
    cos_angles_mask = cos_angles - kpt_cos_angles_mean[None,:,:,:]
    cos_angles_mask = ~torch.isnan(cos_angles_mask)
    ratios_mask = ratios - kpt_ratios_mean[None,:,:,:]
    ratios_mask = ~torch.isnan(ratios_mask)
    signed_volumes_mask = signed_volumes - signed_volumes_mean[None,:]
    signed_volumes_mask = ~torch.isnan(signed_volumes_mask)
    
    cos_angles_likelihood[~cos_angles_mask] = 1
    ratios_likelihood[~ratios_mask] = 1
    signed_volumes_likelihood[~signed_volumes_mask] = 1

    # reshape and take minimum
    cos_angles_likelihood = cos_angles_likelihood.reshape((k*c, -1))
    ratios_likelihood = ratios_likelihood.reshape((k*c, -1))

    cos_angles_likelihood_min = torch.min(cos_angles_likelihood, dim=1).values
    ratios_likelihood_min = torch.min(ratios_likelihood, dim=1).values
    signed_volumes_likelihood_min = torch.min(signed_volumes_likelihood, dim=1).values

    cos_angles_likelihood_min = cos_angles_likelihood_min.reshape((k, c))
    ratios_likelihood_min = ratios_likelihood_min.reshape((k, c))
    signed_volumes_likelihood_min = signed_volumes_likelihood_min.reshape((k, c))
    
    # print("internal model energy")
    # print(cos_angles_likelihood_min[11, 38], cos_angles_likelihood_min[7, 44])
    # print(ratios_likelihood_min[11, 38], cos_angles_likelihood_min[7, 44])
    # print(signed_volumes_likelihood_min[11, 38], cos_angles_likelihood_min[7, 44])


    geom_min_log_likelihood = torch.minimum(cos_angles_likelihood_min, ratios_likelihood_min)
    geom_min_log_likelihood = torch.minimum(geom_min_log_likelihood, signed_volumes_likelihood_min)
    return geom_min_log_likelihood


def select_best_candidate_for_keypoint(
    candidates_for_keypoints: torch.Tensor,
):
    # compute alignment energy and internal model energy
    # take the energy with highest relative distance
    # compare across all fitted candidates
    pass


def select_unlikely_candidates():
    # TODO we neeed energy between candidate and chosen candidates
    # Take minimum likelihood across return values
    internal_model_energy = compute_internal_model_energy()
    pass


def keypoint_candidate_optimization():
    pass


def fit_model(
    kpt_label_likelihood: torch.Tensor,
    candidates_xyz: torch.Tensor,
    preferred_kpt_order: torch.Tensor,
    kpt_threshold: float = 0.8,
    candidate_threshold: float = 0.8,
):
    
    # 0) initialization
    k, n = kpt_label_likelihood.shape
    kpt_chosen_cur = torch.zeros((k,), dtype=torch.bool)
    candidate_chosen_cur = torch.zeros((n,), dtype=torch.bool)
    kpt_indices = torch.arange(k)

    # 1) choose two keypoints 
    # (assumption is that in each image there are at least two stable keypoint indices)
    for i in range(2):
        kpt_index = select_keypoint(
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
        # 1) pick next keypoint
        kpt_index, candidate_index = select_keypoint(
            kpt_likelihood_bias=peak_kpt_likelihood,
            kpt_chosen_cur=kpt_chosen_cur,
            candidate_chosen_cur=candidate_chosen_cur,
            preferred_keypoint_order=preferred_keypoint_order,
            threshold=threshold
        )

        # 2) pick candidates for selected keypoint
        candidates = select_candidates_for_keypoint(kpt_index=kpt_index)

        # 3) optimize for all candidates
        fitted_models = ...


        # 4) select best candidate
        



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


def eval_model():
    pass


def compute_kpt_label_likelihood_for_candidates():
    """
    Update keypoint label likelihood with model energy
    """
    pass



