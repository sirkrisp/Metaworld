
from typing import List, Dict, Callable, Optional, Union
import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
from lightglue import viz2d

from keyframes.align_ft_spair.utils import ft_align_utils
from keyframes.align_ft_spair.ext import utils_correspondence


SPAIR_SORTED_CATEGORIES = [
    'aeroplane', 
    'bicycle', 
    'bird', 
    'boat', 
    'bottle',
    'bus', 
    'car', 
    'cat', 
    'chair', 
    'cow',
    'diningtable', 
    'dog', 
    'horse', 
    'motorbike', 
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

CATEGORY_NAME_TO_PROMPT_NAME = {
    'pottedplant': 'potted plant',
    'tvmonitor': 'tv-monitor'
}


# aeroplane
AEROPLANE_KPT_INDEX_TO_FLIPPED_KPT_INDEX = [0,1,2,3,5,4,7,6,9,8,11,10,13,12,15,14,17,16,19,18,21,20,22,23,24]
AEROPLANE_KPT_INDEX_TO_DESC = {
    0: "nose",
    1: "cockpit",
    2: "forehead",
    3: "landing_gear_front",
    # -- right / left --
    4: "landing_gear_right",
    5: "landing_gear_left",
    6: "engine_front_right",
    7: "engine_front_left",
    8: "wing_end_right",
    9: "wing_end_left",
    10: "engine_back_right",
    11: "engine_back_left",
    12: "wing_foot_front_right",
    13: "wing_foot_front_left",
    14: "wing_foot_back_right",
    15: "wing_foot_back_left",
    16: "tailplane_end_right", # tailplane = horizontal stabilizer
    17: "tailplane_end_left",
    18: "tailplane_foot_front_right",
    19: "tailplane_foot_front_left",
    20: "tailplane_foot_back_right",
    21: "tailplane_foot_back_left",
    # ---------------
    22: "stabilizer_vertical_foot",
    23: "stabilizer_vertical_end",
    24: "rear", # can also be rear engine
}


def aeroplane_get_flipped_kpt_index(kpt_index: int):
    return AEROPLANE_KPT_INDEX_TO_FLIPPED_KPT_INDEX[kpt_index]


def category2prompt_name(category: str):
    """ Returns the prompt name of the category.
    Args:
        category: category name
    """
    return CATEGORY_NAME_TO_PROMPT_NAME.get(category, category)


def category2cat_idx(category: str):
    """ Returns the index of the category in the sorted list of categories.
    Args:
        category: category name
    """
    return SPAIR_SORTED_CATEGORIES.index(category)


def cat_idx2category(cat_idx: int):
    """ Returns the category name of the index in the sorted list of categories.
    Args:
        cat_idx: category index
    """
    return SPAIR_SORTED_CATEGORIES[cat_idx]


def get_cats2files(files: List[str], cats: List[int]) -> Dict[int, List[str]]:
    """ Returns a dictionary mapping each category to a list of files belonging to that category.
    Args:
        files: list of file paths of size 2*N where N is the number of image pairs
        cats: list of category indices of size N where N is the number of image pairs
    Example usage:
        files, kpts, cats, used_points_sets, all_thresholds = utils_dataset.load_and_prepare_data(edict({
            "BBOX_THRE": True,
            "TRAIN_DATASET": None,
            "DATA_ROOT_DIR": "/media/user/ssd2t/datasets2",
            "ANNO_SIZE": 768,
            "SAMPLE": None
        }))
        cats2files = get_cats2files(files, cats)
    """
    cats2files = {}
    for cat in np.unique(cats):
        cats2files[cat] = list(set([files[i] for i in range(len(files)) if cats[i // 2] == cat]))
    return cats2files


def img_file2category(img_filepath: str):
    """ Returns the category of an image file.
    """
    return img_filepath.split("/")[-2]


def load_img(img_filepath, img_size=None, flip=False, pad=False, angle_deg=None):
    img = Image.open(img_filepath)
    if img_size is not None:
        if pad:
            img = utils_correspondence.resize(
                img, 
                target_res=img_size, 
                resize=True, 
                to_pil=True, 
                edge=False
            )
        else:
            # img = utils_correspondence.resize(img, img_size, resize=True, to_pil=True, edge=edge)
            img = img.resize((img_size, img_size))
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if angle_deg is not None:
        # TODO check whether rotatation here works with other functions when padded
        img = img.rotate(angle_deg)
        # TODO  do we need to resize again?
        img = img.resize((img_size, img_size))
    return img


def assemble_all_images(imgs, per_img_size = 48):
    """ Assembles all images of a category into a single image.
    Args:
        cat_idx: category index
        cats2files: dictionary mapping each category to a list of files belonging to that category
    """
    out_img_size = int(np.ceil(len(imgs)**0.5)) * per_img_size
    out_img = np.zeros((out_img_size, out_img_size, 3), dtype=np.uint8)
    for i in range(out_img_size // per_img_size):
        for j in range(out_img_size // per_img_size):
            file_idx = i*(out_img_size // per_img_size)+j
            if file_idx >= len(imgs):
                break
            img_i = imgs[file_idx]
            if type(img_i) == str:
                img = Image.open(img_i)
            else:
                img = Image.fromarray(img_i)
            img = img.resize((per_img_size,per_img_size))
            out_img[i*per_img_size:(i+1)*per_img_size, j*per_img_size:(j+1)*per_img_size,:] = np.array(img)
    return out_img


def load_img_anno(img_filepath: str, spair_data_folder: str):
    """ Load image annotation from a json file.
    Args:
        img_filepath: image file path
        spair_data_folder: path to the SPair dataset
    """
    category = img_file2category(img_filepath)
    img_name_no_ext = os.path.basename(img_filepath).split('.')[0]
    with open(f"{spair_data_folder}/ImageAnnotation/{category}/{img_name_no_ext}.json", 'r') as f:
        img_anno = json.load(f)
    kpts = np.zeros((30,3), dtype=int)
    for kpt_key in img_anno["kps"].keys():
        kpt_coords = img_anno["kps"][kpt_key]
        if kpt_coords is not None:
            kpts[int(kpt_key)] = [*kpt_coords, 1]
    return img_anno, kpts


def get_threshold_from_annotation(img_anno: Dict):
    """ Get the threshold from the annotation.
    Args:
        img_anno: image annotation
        img_size: size of the image
    """
    img_bndbox = img_anno["bndbox"]
    threshold = max(
        (img_bndbox[2] - img_bndbox[0]), # / img_anno["image_width"],
        (img_bndbox[3] - img_bndbox[1]), # / img_anno["image_height"]
    )
    # threshold *= img_size / max(img_anno["image_width"], img_anno["image_height"])
    return threshold


def load_img_embd(img_file: str, embds_folder: str, pad=True, angle_deg=0, flip=0):
    category = img_file2category(img_file)
    img_noext = os.path.basename(img_file).split('.')[0]
    pad_int = 1 if pad else 0
    # if pad:
    #     img_embd_fp = f'{embds_folder}/{category}/{img_noext}.pt'
    # else:
    img_embd_fp = f'{embds_folder}/{category}/{img_noext}_pad={pad_int}_rot={angle_deg}_flip={flip}.pt'
    img_embd = torch.load(img_embd_fp)
    return img_embd


def transform_image_coords(x_orig: int, y_orig: int, img_orig_width: int, img_orig_height: int, img_new_size: int, pad=True):
    """ Transforms image coordinates to new image size. (new image is square)
    Args:
        x_orig: x coordinate in original image
        y_orig: y coordinate in original image
        img_orig_width: width of original image
        img_orig_height: height of original image
        img_new_size: size of new image
        pad: whether to pad the original image to make it square
    """
    # (optional) pad orig image to square image
    img_orig_height_pad, img_orig_width_pad = img_orig_height, img_orig_width
    pad_x_half, pad_y_half = 0, 0
    if pad:
        if img_orig_height < img_orig_width:
            pad_y_half = np.floor((img_orig_width - img_orig_height) / 2)
        elif img_orig_width < img_orig_height:
            pad_x_half = np.floor((img_orig_height - img_orig_width) / 2)
        max_orig_size = max(img_orig_width, img_orig_height)
        img_orig_height_pad = max_orig_size
        img_orig_width_pad = max_orig_size
    x_orig_pad = x_orig + pad_x_half
    y_orig_pad = y_orig + pad_y_half
    # first normalize and then resize to new image size
    x_new = int(np.floor(x_orig_pad / img_orig_width_pad * img_new_size))
    y_new = int(np.floor(y_orig_pad / img_orig_height_pad * img_new_size))
    return x_new, y_new


def transform_image_coords_parallel(
    img_coords: Union[torch.Tensor, np.ndarray],
    img_orig_width: int,
    img_orig_height: int,
    img_new_size: int,
):
    """
    Args:
        - img_coords: (n, 2)
        - img_orig_width: int
        - img_orig_height: int
        - img_new_size: int
    """
    embd_coords_xy = []
    for i in range(len(img_coords)):
        x_new, y_new = transform_image_coords(
            x_orig=int(img_coords[i,0]),
            y_orig=int(img_coords[i,1]),
            img_orig_width=img_orig_width,
            img_orig_height=img_orig_height,
            img_new_size=img_new_size,
            pad=True
        )
        embd_coords_xy.append([x_new, y_new])
    embd_coords_xy_torch = torch.tensor(embd_coords_xy)
    return embd_coords_xy_torch


def transform_image_coords_inv(x_new, y_new, img_orig_width, img_orig_height, img_new_size, pad=True):
    img_orig_height_pad, img_orig_width_pad = img_orig_height, img_orig_width
    pad_x_half, pad_y_half = 0, 0
    if pad:
        if img_orig_height < img_orig_width:
            pad_y_half = np.floor((img_orig_width - img_orig_height) / 2)
        elif img_orig_width < img_orig_height:
            pad_x_half = np.floor((img_orig_height - img_orig_width) / 2)
        max_orig_size = max(img_orig_width, img_orig_height)
        img_orig_height_pad = max_orig_size
        img_orig_width_pad = max_orig_size
    x_orig_pad = int(np.floor(x_new / img_new_size * img_orig_width_pad))
    y_orig_pad = int(np.floor(y_new / img_new_size * img_orig_height_pad))
    x_orig = x_orig_pad - pad_x_half
    y_orig = y_orig_pad - pad_y_half
    return x_orig, y_orig


def load_kpt_data(img_files, spair_data_folder, embds_folder, img_new_size=768, embd_size=48, angles_deg=[0], pad=False):
    kpt_data = {i: [] for i in range(30)}
    for img_file in img_files:
        img_anno, img_kpts = load_img_anno(img_file, spair_data_folder)
        for angle in angles_deg:
            # load img embedding
            img_embd = load_img_embd(img_file, embds_folder, pad=pad, angle_deg=angle)
            for kpt_idx in range(30):
                if img_kpts[kpt_idx][2] > 0:
                    x_orig, y_orig = img_kpts[kpt_idx][:2]
                    img_orig_width, img_orig_height = img_anno["image_width"], img_anno["image_height"]
                    x_resized, y_resized = transform_image_coords(x_orig, y_orig, img_orig_width, img_orig_height, img_new_size, pad)
                    kpt_embd, x_embd, y_embd = ft_align_utils.get_img_embd_at_xy_orig(
                        x=x_resized,
                        y=y_resized,
                        embd=img_embd[0],
                        img_w=img_new_size,
                        embd_w=embd_size,
                        angle_deg=angle,
                        flip=0
                    )
                    # if image is rotated kpt might be out of view
                    if kpt_embd is not None:
                        # kpt coords to embedding patch
                        kpt_data[kpt_idx].append({
                            "kpt": list(img_kpts[kpt_idx][:2]),
                            "kpt_embd_coords": [x_embd, y_embd],
                            "kpt_embedding": kpt_embd,
                            "img": img_file
                        })
    return kpt_data


def load_kpt_embd_coords(
    img_files: List[str],
    spair_data_folder: str,
    img_new_size=768,
    embd_size=48,
    angle_deg=0,
    flip=0,
    pad=False,
    get_flipped_kpt_index: Optional[Callable[[int], int]]=None
):
    assert flip == 0 or (flip == 1 and get_flipped_kpt_index is not None), "if flip is 1, get_flipped_kpt_index must be set"
    img_kpt_embd_coords = []
    all_kpt_img_coords = []
    for img_file in img_files:
        kpt_embd_coords = []
        kpt_img_coords = []
        img_anno, img_kpts = load_img_anno(img_file, spair_data_folder)
        for kpt_idx in range(30):
            if img_kpts[kpt_idx][2] > 0:
                w, h = img_anno["image_width"], img_anno["image_height"]
                x_img, y_img = img_kpts[kpt_idx][:2]
                # TODO when transforming keypoints, we actually do not need to know the size of the new image
                # scale and rotate keypoint
                x, y = transform_image_coords(x_img, y_img, w, h, img_new_size, pad)
                x, y = ft_align_utils.transform_point(x, y, img_new_size, flip=flip, angle_deg=angle_deg)
                kpt_index_transformed = kpt_idx
                if flip == 1:
                    kpt_index_transformed = get_flipped_kpt_index(kpt_idx)
                # project keypoint coords to embedding coords
                x_embd, y_embd = ft_align_utils.img_to_embedding_coords(x, y, img_w=img_new_size, embd_w=embd_size)
                if x_embd >= 0 and x_embd < embd_size and y_embd >= 0 and y_embd < embd_size:
                    kpt_embd_coords.append([x_embd, y_embd, kpt_index_transformed])
                    # TODO make it work for angle
                    x_img, y_img = ft_align_utils.transform_point(x_img, y_img, w, flip=flip, angle_deg=0)
                    kpt_img_coords.append([x_img, y_img, kpt_index_transformed])
        img_kpt_embd_coords.append(torch.tensor(kpt_embd_coords))
        all_kpt_img_coords.append(torch.tensor(kpt_img_coords))
    return img_kpt_embd_coords, all_kpt_img_coords


def concatenate_and_stack_kpt_embeddings(kpt_data):
    # 1) concatenate embeddings for each kpt index
    kptIdx2kpt_embds = {}
    max_imgs, embd_dim = 0, 0
    for kpt_idx in range(30):
        if len(kpt_data[kpt_idx]) > 0:
            kptIdx2kpt_embds[kpt_idx] = torch.cat([kpt["kpt_embedding"] for kpt in kpt_data[kpt_idx]], dim=0)
            max_imgs = max(max_imgs, len(kpt_data[kpt_idx]))
            embd_dim = kptIdx2kpt_embds[kpt_idx].shape[-1]
    # 2) stack (and pad) concatenated kpt embeddings
    stacked_kpt_embds = torch.zeros((30, max_imgs, embd_dim))
    for kpt_idx in kptIdx2kpt_embds.keys():
        kpt_embds = kptIdx2kpt_embds[kpt_idx]
        stacked_kpt_embds[kpt_idx, :kpt_embds.shape[0], :] = kpt_embds
    return stacked_kpt_embds


def evaluate_spair(
    img_files: List[int],
    sim_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    spair_data_folder: str,
    embds_folder: Optional[str] = None,
    embds_loader: Optional[Callable[[str], torch.Tensor]] = None,
    img_size=768,
    embd_size=48,
    pad=True,
    allow_left_right=False,
    left_right_pairs=[]
):
    """ Evaluate SPair dataset.
    Args:
        img_files: list of image files of size 2*N, where N is the number of image pairs 
            and (2*i, 2*i+1) are the images of the i-th pair
        sim_fun: similarity function that takes query embd and img embd and returns similarity between query and img embds
        spair_data_folder: path to the SPair dataset
        embds_folder: path to the folder containing image embeddings
        embds_loader: function that loads image embeddings given the image file path
        img_size: size of the input image
        embd_size: size of the input embedding
    """
    # TODO instead of loading each image json file for each image we should use the pair annotation
    assert embds_folder is not None or embds_loader is not None, "either embds_folder or embds_loader must be set"
    N = len(img_files) // 2
    total_num_kpts = 0
    alphas = [0.1, 0.05, 0.01]
    alpha_statistics = {alpha: 0 for alpha in alphas}
    pair_statistics = {f"acc_{alpha}": [] for alpha in alphas}

    for pair_idx in tqdm(range(N)):
        # Load image annotations and extract keypoints and thresholds
        img1_anno, img1_kps = load_img_anno(img_files[2*pair_idx], spair_data_folder)
        img2_anno, img2_kps = load_img_anno(img_files[2*pair_idx+1], spair_data_folder)
        h1, w1 = img1_anno["image_height"], img1_anno["image_width"]
        h2, w2 = img2_anno["image_height"], img2_anno["image_width"]
        threshold = get_threshold_from_annotation(img2_anno)

        is_match = img1_kps[:, 2] * img2_kps[:, 2] > 0
        # img1_kps_all = img1_kps
        img2_kps_all = img2_kps
        img1_kps = img1_kps[is_match, :2]
        img2_kps = img2_kps[is_match, :2]
        kpt_indices = np.arange(30)[is_match]

        # load embeddings
        if embds_loader is None:
            img_embd1 = load_img_embd(img_files[2*pair_idx], embds_folder, pad=pad)
            img_embd2 = load_img_embd(img_files[2*pair_idx+1], embds_folder, pad=pad)
        else:
            with torch.no_grad():
                img_embd1 = embds_loader(img_files[2*pair_idx])
                img_embd2 = embds_loader(img_files[2*pair_idx+1])

        # for each keypoint in img1, find the corresponding location in img2
        total_num_kpts += img1_kps.shape[0]
        pair_correct_counts = {alpha: 0 for alpha in alphas}
        for j in range(img1_kps.shape[0]):
            x_src, y_src = img1_kps[j, :2]
            x_trg, y_trg = img2_kps[j, :2]

            # transform keypoints to resized image used as input to the image encoder
            x_src_resized, y_src_resized = transform_image_coords(x_src, y_src, w1, h1, img_size, pad=pad)
            query_embd, _, _ = ft_align_utils.get_img_embd_at_xy_orig(
                x=x_src_resized, y=y_src_resized, embd=img_embd1[0], img_w=img_size, embd_w=embd_size, angle_deg=0, flip=0
            )
            with torch.no_grad():
                similarity = sim_fun(query_embd, img_embd2)
            # # upsample dist to original image size
            # similarity = nn.Upsample(size=(h2, w2), mode='bilinear')(similarity[None, None, :, :])[0,0]  # shape (H, W)
            # upsample dist to image input size
            similarity = nn.Upsample(size=(img_size, img_size), mode='bilinear')(similarity[None, None, :, :])[0,0]  # shape (img_size, img_size)
            similarity = similarity.cpu().numpy()
            max_yx = np.unravel_index(similarity.argmax(), similarity.shape)
            x_pred, y_pred = transform_image_coords_inv(max_yx[1], max_yx[0], w2, h2, img_size, pad=pad)

            dist = ((x_pred - x_trg) ** 2 + (y_pred - y_trg) ** 2) ** 0.5

            if allow_left_right:
                kpt_index = kpt_indices[j]
                if kpt_index in left_right_pairs:
                    kpt_index_other = kpt_index
                    if kpt_index % 2 == 0:
                        kpt_index_other += 1
                    else:
                        kpt_index_other -= 1
                    x_trg_other, y_trg_other, exists = img2_kps_all[kpt_index_other, :3]
                    if exists > 0:
                        dist_other = ((x_pred - x_trg_other) ** 2 + (y_pred - y_trg_other) ** 2) ** 0.5
                        dist = min(dist, dist_other)
                    else:
                        print("other kpt_index does not exist", kpt_index, kpt_index_other)

            for alpha in alphas:
                if dist <= (alpha * threshold):
                    alpha_statistics[alpha] += 1
                    pair_correct_counts[alpha] += 1
        for alpha in alphas:
            # print(alpha, pair_correct_counts[alpha] / img1_kps.shape[0])
            pair_statistics[f'acc_{alpha}'].append(pair_correct_counts[alpha] / img1_kps.shape[0])
    for alpha in alphas:
        alpha_statistics[alpha] /= total_num_kpts
        pair_statistics[f'acc_{alpha}_mean'] = np.mean(pair_statistics[f'acc_{alpha}']) * 100
    return alpha_statistics, pair_statistics


def evaluate_spair_v2(
    img_files: List[int],
    sim_fun: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    spair_data_folder: str,
    embds_loader: Optional[Callable[[str], torch.Tensor]],
    img_size=768,
    embd_size=48,
    pad=True,
):
    """ Evaluate SPair dataset.
    Args:
        img_files: list of image files of size 2*N, where N is the number of image pairs 
            and (2*i, 2*i+1) are the images of the i-th pair
        sim_fun: similarity function that takes query embd and img embd and returns similarity between query and img embds
        spair_data_folder: path to the SPair dataset
        embds_folder: path to the folder containing image embeddings
        embds_loader: function that loads image embeddings given the image file path
        img_size: size of the input image
        embd_size: size of the input embedding
    """
    # TODO instead of loading each image json file for each image we should use the pair annotation
    N = len(img_files) // 2
    total_num_kpts = 0
    alphas = [0.1, 0.05, 0.01]
    alpha_statistics = {alpha: 0 for alpha in alphas}
    pair_statistics = {f"acc_{alpha}": [] for alpha in alphas}

    for pair_idx in tqdm(range(N)):
        # Load image annotations and extract keypoints and thresholds
        img1_anno, img1_kps = load_img_anno(img_files[2*pair_idx], spair_data_folder)
        img2_anno, img2_kps = load_img_anno(img_files[2*pair_idx+1], spair_data_folder)
        h1, w1 = img1_anno["image_height"], img1_anno["image_width"]
        h2, w2 = img2_anno["image_height"], img2_anno["image_width"]
        threshold = get_threshold_from_annotation(img2_anno)

        is_match = img1_kps[:, 2] * img2_kps[:, 2] > 0
        img1_kps = img1_kps[is_match, :2]
        img2_kps = img2_kps[is_match, :2]

        # load embeddings
        with torch.no_grad():
            img_embd1 = embds_loader(img_files[2*pair_idx])
            img_embd2 = embds_loader(img_files[2*pair_idx+1])

        # for each keypoint in img1, find the corresponding location in img2
        total_num_kpts += img1_kps.shape[0]
        pair_correct_counts = {alpha: 0 for alpha in alphas}
        
        query_embd_coords = []
        for j in range(img1_kps.shape[0]):
            x_src, y_src = img1_kps[j, :2]
            x_trg, y_trg = img2_kps[j, :2]
            # transform keypoints to resized image used as input to the image encoder
            x_src_resized, y_src_resized = transform_image_coords(x_src, y_src, w1, h1, img_size, pad=pad)
            query_embd, x_embd, y_embd = ft_align_utils.get_img_embd_at_xy_orig(
                x=x_src_resized, y=y_src_resized, embd=img_embd1[0], img_w=img_size, embd_w=embd_size, angle_deg=0, flip=0
            )
            if query_embd is not None:
                query_embd_coords.append([x_embd, y_embd])
        query_embd_coords = torch.tensor(query_embd_coords)

        with torch.no_grad():
            kpt_sims = sim_fun(query_embd_coords, img_embd1, img_embd2)
        
        for j in range(img1_kps.shape[0]):
            similarity = kpt_sims[j]
            # # upsample dist to original image size
            # similarity = nn.Upsample(size=(h2, w2), mode='bilinear')(similarity[None, None, :, :])[0,0]  # shape (H, W)
            # upsample dist to image input size
            similarity = nn.Upsample(size=(img_size, img_size), mode='bilinear')(similarity[None, None, :, :])[0,0]  # shape (img_size, img_size)
            similarity = similarity.cpu().numpy()
            max_yx = np.unravel_index(similarity.argmax(), similarity.shape)
            x_pred, y_pred = transform_image_coords_inv(max_yx[1], max_yx[0], w2, h2, img_size, pad=pad)

            dist = ((x_pred - x_trg) ** 2 + (y_pred - y_trg) ** 2) ** 0.5
            for alpha in alphas:
                if dist <= (alpha * threshold):
                    alpha_statistics[alpha] += 1
                    pair_correct_counts[alpha] += 1
        for alpha in alphas:
            # print(alpha, pair_correct_counts[alpha] / img1_kps.shape[0])
            pair_statistics[f'acc_{alpha}'].append(pair_correct_counts[alpha] / img1_kps.shape[0])
    for alpha in alphas:
        alpha_statistics[alpha] /= total_num_kpts
        pair_statistics[f'acc_{alpha}_mean'] = np.mean(pair_statistics[f'acc_{alpha}']) * 100
    return alpha_statistics, pair_statistics


def visualize_prediction(
    img1_file: str,
    img2_file: str,
    sim_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
    spair_data_folder: str, 
    embds_folder: Optional[str] = None,
    embds_loader: Optional[Callable[[str], torch.Tensor]] = None,
    img_size = 768,
    embd_size = 48,
    pad=False
):
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)
    # Load image annotations and extract keypoints and thresholds
    img1_anno, img1_kps = load_img_anno(img1_file, spair_data_folder)
    img2_anno, img2_kps = load_img_anno(img2_file, spair_data_folder)
    h1, w1 = img1_anno["image_height"], img1_anno["image_width"]
    h2, w2 = img2_anno["image_height"], img2_anno["image_width"]

    is_match = img1_kps[:, 2] * img2_kps[:, 2] > 0
    kpt_indices = np.arange(0,30,1)[is_match]
    img1_kps = img1_kps[is_match, :2]
    img2_kps = img2_kps[is_match, :2]

    # load embeddings
    if embds_folder is not None:
        img_embd1 = load_img_embd(img1_file, embds_folder, pad=pad)
        img_embd2 = load_img_embd(img2_file, embds_folder, pad=pad)
    elif embds_loader is not None:
        with torch.no_grad():
            img_embd1 = embds_loader(img1_file)
            img_embd2 = embds_loader(img2_file)

    # pred kps
    img2_pred_kps = []
    similarities = []
    for j in range(img1_kps.shape[0]):
        x_src, y_src = img1_kps[j, :2]
        # x_trg, y_trg = img2_kps[j, :2]

        # transform keypoints to resized image used as input to the image encoder
        x_src_resized, y_src_resized = transform_image_coords(x_src, y_src, w1, h1, img_size, pad=pad)
        query_embd, _, _ = ft_align_utils.get_img_embd_at_xy_orig(
            x=x_src_resized, y=y_src_resized, embd=img_embd1[0], img_w=img_size, embd_w=embd_size, angle_deg=0, flip=0
        )
        with torch.no_grad():
            similarity = sim_fun(query_embd, img_embd2)
        similarity = nn.Upsample(size=(img_size, img_size), mode='bilinear')(similarity[None, None, :, :])[0,0]  # shape (img_size, img_size)
        similarity = similarity.cpu().numpy()
        max_yx = np.unravel_index(similarity.argmax(), similarity.shape)
        x_pred, y_pred = transform_image_coords_inv(max_yx[1], max_yx[0], w2, h2, img_size, pad=pad)
        # upsample dist to original image size
        # similarity = nn.Upsample(size=(h2, w2), mode='bilinear')(similarity[None, None, :, :])[0,0]  # shape (H, W)
        # similarity = similarity.cpu().numpy()
        # similarities.append(similarity)
        # max_yx = np.unravel_index(similarity.argmax(), similarity.shape)

        img2_pred_kps.append([x_pred, y_pred])
    img2_pred_kps = np.array(img2_pred_kps)
    viz2d.plot_images([np.array(img1), np.array(img2)])
    viz2d.plot_matches(img1_kps, img2_pred_kps)
    return [img1, img2], [img1_kps, img2_kps, img2_pred_kps, kpt_indices], similarities

    

def build_kpt_idx_to_kpt_embds(
    img_files: List[str],
    img_embds_hat: torch.Tensor,
    spair_data_folder: str,
    img_size=960,
    embd_size=60,
    pad=True,
    flips=[False]
):  
    # extract embeddings at embedding coords of keypoints
    kpt_embd_coords: List[torch.Tensor] = []
    kpt_img_coords: List[torch.Tensor] = []
    for flip in flips:
        emb_coords, img_coords = load_kpt_embd_coords(
            img_files=img_files,
            spair_data_folder=spair_data_folder,
            img_new_size=img_size,
            embd_size=embd_size,
            angle_deg=0,
            flip=1 if flip else 0,
            pad=pad,
            # TODO needs to be adjusted for other cetgories!!!
            get_flipped_kpt_index=aeroplane_get_flipped_kpt_index,
        )
        kpt_embd_coords += emb_coords
        kpt_img_coords += img_coords

    # some basic check
    if len(flips) == 2:
        n_imgs = len(img_files)
        for i in range(n_imgs):
            if (
                kpt_embd_coords[i].shape[0]
                != kpt_embd_coords[i + n_imgs].shape[0]
            ):
                print(i, kpt_embd_coords[i].shape)

    # collect features at embeddings coords of keypoints
    max_n_kpts = 30
    kpt_idx_to_kpt_embds_list: Dict[int,List[torch.Tensor]] = {i: [] for i in range(max_n_kpts)}
    for i, embd_coords in enumerate(kpt_embd_coords):
        if embd_coords.size(0) > 0:
            kpt_embds = img_embds_hat[
                i, :, embd_coords[:, 1], embd_coords[:, 0]
            ]  # shape (C, n_kpts)
            for j, kpt_idx in enumerate(embd_coords[:, 2].tolist()):
                kpt_idx_to_kpt_embds_list[kpt_idx].append(kpt_embds[:, j])
    # kpt_idx_to_kpt_embds[kpt_index] = stacked features corresponding to keypoints with kpt_index
    kpt_idx_to_kpt_embds: Dict[int, Union[torch.Tensor, None]] = {}
    for i in range(max_n_kpts):
        if len(kpt_idx_to_kpt_embds_list[i]) > 0:
            kpt_idx_to_kpt_embds[i] = torch.stack(kpt_idx_to_kpt_embds_list[i])
        else:
            kpt_idx_to_kpt_embds[i] = None
    return kpt_idx_to_kpt_embds, kpt_embd_coords, kpt_img_coords


def select_images_with_keypoint_indices(include_kpt_indices, exclude_kpt_indices, kpt_img_coords):
    selected_img_indices = []
    include_kpt_indices_set = set(include_kpt_indices)
    exclude_kpt_indices_set = set(exclude_kpt_indices)
    for i in range(len(kpt_img_coords)):
        kpt_img_coords_i = kpt_img_coords[i]
        kpt_indices_in_img = kpt_img_coords_i[:,2]
        kpt_indices_in_img_set = set(kpt_indices_in_img.tolist())
        if include_kpt_indices_set.intersection(kpt_indices_in_img_set) == include_kpt_indices_set and exclude_kpt_indices_set.intersection(kpt_indices_in_img_set) == set():
            selected_img_indices.append(i)
    return selected_img_indices



