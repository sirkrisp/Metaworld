from typing import Optional
import torch
import numpy as np
from keyframes.align_ft_spair.utils import ft_align_utils, spair_utils, geom_utils, kpt_likelihood_utils
from utils import torch_utils
from tqdm import tqdm




def assign_keypoint(
    kpt_likelihood_bias: torch.Tensor,
    candidates_xyz: torch.Tensor,
    kpt_xyz_cur: torch.Tensor,
    kpt_xyz_ref: torch.Tensor,
    kpt_chosen_cur: torch.Tensor,
    candidate_chosen_cur: torch.Tensor,

    # params
    kpt_is_similar_threshold=0.4,
    kpt_is_similar_threshold_energy=0.4,
    kpt_min_likelihood=0.6,
):
    """
    Args:
        - kpt_likelihood_bias: (K, N) tensor with likelihood of each keypoint at each candidate location
        - candidates_xyz: (N, 3) tensor with candidate xyz locations
        - kpt_xyz_cur: (K, 3) tensor with current keypoint xyz locations
        - kpt_xyz_ref: (K, 3) tensor with reference keypoint xyz locations
        - kpt_chosen_cur: (K,) tensor with boolean mask of chosen keypoints
        - candidate_chosen_cur: (N,) tensor with boolean mask of chosen candidates
        - kpt_is_similar_threshold: float, threshold for similarity between keypoints 
            (keypoint is considered similar if its likelihood is 1/(1+kpt_is_similar_threshold) of the maximum likelihood, 
            that is, if the likelihood of the keypoints with maximum likelihood is kpt_is_similar_threshold more likely 
            than all other keypoints, then there are no similar keypoints)
        - kpt_is_similar_threshold_energy: like kpt_is_similar_threshold, but based on energy instead of likelihood
        - kpt_min_likelihood: float, minimum likelihood for keypoint to be considered
    """

    k, n = kpt_likelihood_bias.shape
    not_chosen = torch.logical_not(kpt_chosen_cur)
    not_chosen_candidates = torch.logical_not(candidate_chosen_cur)
    kpt_indices = torch.arange(k)
    candidate_indices = torch.arange(n)
    kpt_likelihood_bias_of_not_chosen = kpt_likelihood_bias[not_chosen,not_chosen_candidates]
    kpt_indices_not_chosen = kpt_indices[not_chosen]
    candidate_indices_not_chosen = candidate_indices[not_chosen_candidates]

    if kpt_likelihood_bias_of_not_chosen.numel() == 0:
        return kpt_xyz_cur.clone(), kpt_chosen_cur.clone(), candidate_chosen_cur.clone()
    
    # TODO we should first compute kpt_ambiguity likelihood (combine likelihood and energy with max), 
    #   then multiply with likelihood_bias to retrieve max likelihood
    # 1) get candidate xyz with maximum likelihood
    kpt_total_likelihood_argmax_not_chosen = torch.argmax(kpt_likelihood_bias_of_not_chosen, dim=-1) # (K,)
    kpt_max_likelihoods_not_chosen = kpt_likelihood_bias_of_not_chosen[:, kpt_total_likelihood_argmax_not_chosen] # (K,)
    sort_res_not_chosen = torch.sort(kpt_max_likelihoods_not_chosen, descending=True)
    kpt_index_in_not_chosen = sort_res_not_chosen.indices[0]
    kpt_index = kpt_indices_not_chosen[kpt_index_in_not_chosen]
    candidate_index_in_not_chosen = kpt_total_likelihood_argmax_not_chosen[kpt_index_in_not_chosen]
    candidate_index = candidate_indices_not_chosen[candidate_index_in_not_chosen]

    # 1.2) if kpt_likelihood is below threshold, return
    if kpt_max_likelihoods_not_chosen[0] < kpt_min_likelihood:
        return kpt_xyz_cur.clone(), kpt_chosen_cur.clone(), candidate_chosen_cur.clone()

    # 2) get similar keypoints to kpt_index at candidate location
    # TODO there should be also an ambiguity term across candidate locations
    # => We could also do this in post-processing, refine assignment to other candidate location after all keypoints have been assigned
    kpt_likelihood_bias_not_chosen_at_candidate = kpt_likelihood_bias_of_not_chosen[:,candidate_index_in_not_chosen]
    kpt_ambiguity_likelihood_not_chosen = kpt_likelihood_bias_not_chosen_at_candidate / (torch.sum(kpt_likelihood_bias_not_chosen_at_candidate, dim=0, keepdim=True) + 1e-6)
    similar_kpt_indices_mask = torch.logical_and(
        kpt_ambiguity_likelihood_not_chosen[kpt_index_in_not_chosen] > 1/(1+kpt_is_similar_threshold),
        candidate_indices_not_chosen != kpt_index
    )
    similar_kpt_indices = candidate_indices_not_chosen[similar_kpt_indices_mask]

    # 3) evaluate energy at candidate location which might disambiguate the similar keypoints
    if len(similar_kpt_indices) > 0:
        model_energy_not_chosen_at_candidate = kpt_likelihood_utils.eval_energy_for_query_xyz(
            kpt_xyz_cur, kpt_chosen_cur, candidates_xyz[candidate_index,:], kpt_xyz_ref
        ) # (#not_chosen, 1)
        model_energy_likelihood_not_chosen = torch.exp(-model_energy_not_chosen_at_candidate).float()
        kpt_energy_ambiguity_not_chosen = model_energy_likelihood_not_chosen / (torch.sum(model_energy_likelihood_not_chosen, dim=0, keepdim=True) + 1e-6)
        similar_kpt_indices_mask = torch.logical_and(
            kpt_energy_ambiguity_not_chosen[kpt_index_in_not_chosen] > 1/(1+kpt_is_similar_threshold_energy),
            similar_kpt_indices_mask
        )
        similar_kpt_indices = candidate_indices_not_chosen[similar_kpt_indices_mask]

    # 4) if there are no similar keypoints, return kpt_index
    if len(similar_kpt_indices) == 0:
        kpt_xyz_cur[kpt_index] = candidates_xyz[candidate_index]
        kpt_chosen_cur[kpt_index] = True
        candidate_chosen_cur[candidate_index] = True
        return assign_keypoint(
            kpt_likelihood_bias=kpt_likelihood_bias,
            candidates_xyz=candidates_xyz,
            kpt_xyz_cur=kpt_xyz_cur,
            kpt_xyz_ref=kpt_xyz_ref,
            kpt_chosen_cur=kpt_chosen_cur,
            candidate_chosen_cur=candidate_chosen_cur,

            # params
            kpt_is_similar_threshold=kpt_is_similar_threshold,
            kpt_is_similar_threshold_energy=kpt_is_similar_threshold_energy,
        )
    
    # 5) evaluate energy for each similar keypoint
    n_similar = similar_kpt_indices.shape[0]
    minimum_energy = torch.inf
    prev_kpt_xyz_cur = kpt_xyz_cur.clone()
    prev_kpt_chosen_cur = kpt_chosen_cur.clone()
    prev_candidate_chosen_cur = candidate_chosen_cur.clone()
    for i in range(n_similar):
        new_kpt_index = similar_kpt_indices[i]
        kpt_xyz_cur = prev_kpt_xyz_cur.clone()
        kpt_chosen_cur = prev_kpt_chosen_cur.clone()
        candidate_chosen_cur = prev_candidate_chosen_cur.clone()
        kpt_xyz_cur[kpt_index] = candidates_xyz[new_kpt_index]
        kpt_chosen_cur[new_kpt_index] = True
        candidate_chosen_cur[candidate_index] = True
        
        # TODO do not go deeper than 2 levels, then, after this for loop, return assign_keypoint with no level limitation
        kpt_xyz_cur, kpt_chosen_cur, candidate_chosen_cur = assign_keypoint(
            kpt_likelihood_bias=kpt_likelihood_bias,
            candidates_xyz=candidates_xyz,
            kpt_xyz_cur=kpt_xyz_cur,
            kpt_xyz_ref=kpt_xyz_ref,
            kpt_chosen_cur=kpt_chosen_cur,
            candidate_chosen_cur=candidate_chosen_cur,

            # params
            kpt_is_similar_threshold=kpt_is_similar_threshold,
            kpt_is_similar_threshold_energy=kpt_is_similar_threshold_energy,
        )
        # all keypoints should be assigned at this stage
        energy_i = kpt_likelihood_utils.eval_energy_for_chosen(
            kpt_xyz_cur, kpt_chosen_cur, kpt_xyz_ref
        )

        if energy_i < minimum_energy:
            minimum_energy = energy_i

    return kpt_xyz_cur, kpt_chosen_cur, candidate_chosen_cur