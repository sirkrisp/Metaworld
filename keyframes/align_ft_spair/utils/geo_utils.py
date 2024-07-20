from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import os
from keyframes.align_ft_spair.utils import spair_utils


def min_max_norm(ft, eps=1e-10):
    norms = torch.linalg.norm(ft, dim=1, keepdim=True)
    ft = ft / (norms + eps)
    return ft

def get_geo_embd(img_filepath: str, embds_folder_dino: str, embds_folder_sd: str, flip=False, num_patches = 60, enable_min_max_norm=True, min_max_norm_epsilon=1e-10, alpha=0.5):
    """ Load geo embeddings.
    Returns:
        ft_geo: tensor of shape (1, C, num_patches, num_patches), C = 3328
    """
    img_filename = os.path.basename(img_filepath).split('.')[0]
    category = spair_utils.img_file2category(img_filepath)
    flip_suffix = '_flip' if flip else ''
    # TODO rename dino embds to standard format so we can use spair_utils.load_img_embd
    pt_dino = f'{embds_folder_dino}/{category}/{img_filename}_dino{flip_suffix}.pt'
    ft_dino = torch.load(pt_dino).to("cuda").detach()
    
    # pt_sd = f'{geo_embds_folder}/{category}/{img_filename}_sd{flip_suffix}.pt'
    # ft_sd = torch.load(pt_sd)
    ft_sd = spair_utils.load_img_embd(img_filepath, embds_folder_sd, pad=True, angle_deg=0, flip=int(flip)).to("cuda").detach()
    # interpolate
    ft_dino = F.interpolate(ft_dino, size=(num_patches, num_patches), mode='bilinear', align_corners=False)
    # ft_sd = torch.cat([
    #     # sd_features['s3'],
    #     F.interpolate(ft_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
    #     F.interpolate(ft_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
    # ], dim=1)
    ft_sd = F.interpolate(ft_sd, size=(num_patches, num_patches), mode='bilinear', align_corners=False)
    # normalize
    if enable_min_max_norm: # NOTE min max norm makes basically no difference
        ft_dino = min_max_norm(ft_dino, eps=min_max_norm_epsilon)
        ft_sd = min_max_norm(ft_sd, eps=min_max_norm_epsilon)
    ft_geo = torch.cat([alpha*ft_sd, (1-alpha)*ft_dino], dim=1)
    # ft_geo = ft_dino
    return ft_geo.detach()


def get_geo_embd_aggregated(img_filepath: str, embds_folder_geo: str, aggre_net: nn.Module, flip=False, num_patches = 60):
    """ Load geo embeddings.
    Returns:
        ft_geo: tensor of shape (1, C, num_patches, num_patches), C = 3328
    """
    with torch.no_grad():
        img_filename = os.path.basename(img_filepath).split('.')[0]
        category = spair_utils.img_file2category(img_filepath)
        flip_suffix = '_flip' if flip else ''
        # TODO rename dino embds to standard format so we can use spair_utils.load_img_embd

        # dino feature
        pt_dino = f'{embds_folder_geo}/{category}/{img_filename}_dino{flip_suffix}.pt'
        ft_dino = torch.load(pt_dino).to("cuda").detach()
        ft_dino = F.interpolate(ft_dino, size=(num_patches, num_patches), mode='bilinear', align_corners=False)
        
        # diffusion features
        pt_sd = f'{embds_folder_geo}/{category}/{img_filename}_sd{flip_suffix}.pt'
        ft_sd = torch.load(pt_sd)

        # aggregate the features and apply post-processing
        desc_gathered = torch.cat([
            ft_sd['s3'],
            F.interpolate(ft_sd['s4'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            F.interpolate(ft_sd['s5'], size=(num_patches, num_patches), mode='bilinear', align_corners=False),
            ft_dino
        ], dim=1)
        desc = aggre_net(desc_gathered) # 1, 768, 60, 60

    # normalize the descriptors
    norms_desc = torch.linalg.norm(desc, dim=1, keepdim=True)
    desc = desc / (norms_desc + 1e-8)
    return desc


def load_geo_embds(img_files: List[str], embds_folder_dino: str, embds_folder_sd: str, flips=[False]):
    img_embds = []
    for flip in flips:
        for img_file in img_files:
            img_embds.append(get_geo_embd(
                img_file, 
                embds_folder_dino=embds_folder_dino, 
                embds_folder_sd=embds_folder_sd, 
                flip=flip
            ))
    img_embds = torch.cat(img_embds, dim=0)
    img_embds_hat = F.normalize(img_embds)
    return img_embds, img_embds_hat

def load_geo_embds_aggregated(img_files: List[str], embds_folder_geo: str, aggre_net: nn.Module, flips=[False]):
    img_embds = []
    for flip in flips:
        for img_file in img_files:
            img_embds.append(get_geo_embd_aggregated(
                img_file, 
                embds_folder_geo=embds_folder_geo, 
                aggre_net=aggre_net,
                flip=flip
            ))
    img_embds = torch.cat(img_embds, dim=0)
    img_embds_hat = F.normalize(img_embds)
    return img_embds, img_embds_hat