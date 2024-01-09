from gc import disable
from typing import Callable, List, Union, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import utils.torch_utils as torch_utils
import torchvision.transforms.functional as tF
import cv2


def scale_coords(x, y, img_shape, model_res):
    h, w, c = img_shape
    if h < w:
        resized_h = model_res
        resized_w = int(w / h * model_res)
    else:
        resized_w = model_res
        resized_h = int(h / w * model_res)
    return int(x / w * resized_w), int(y / h * resized_h)


def img_coords_orig_to_resized_np(x, y, orig_img_shape, model_res):
    h, w, c = orig_img_shape
    if h < w:
        resized_h = model_res
        resized_w = int(w / h * model_res)
    else:
        resized_w = model_res
        resized_h = int(h / w * model_res)
    return np.floor(x / w * resized_w).astype(int), np.floor(y / h * resized_h).astype(int)


def img_coords_resized_to_orig(x, y, orig_img_shape, model_res):
    h, w, c = orig_img_shape
    if h < w:
        resized_h = model_res
        resized_w = int(w / h * model_res)
    else:
        resized_w = model_res
        resized_h = int(h / w * model_res)
    return int(x / resized_w * w), int(y / resized_h * h)


def img_coords_resized_to_orig_np(x: np.ndarray, y: np.ndarray, orig_img_shape, model_res):
    h, w, c = orig_img_shape
    if h < w:
        resized_h = model_res
        resized_w = int(w / h * model_res)
    else:
        resized_w = model_res
        resized_h = int(h / w * model_res)
    return np.floor(x / resized_w * w).astype(int), np.floor(y / resized_h * h).astype(int)


def query_embedding(query_embd: torch.Tensor, img_embd: torch.Tensor):
    """
    Args:
        query_embd: (c)
        img_embd: (b, h, w, c)

    Returns:
        attn: (b, h, w)
    """
    with torch.no_grad():
        b, h, w, c = img_embd.shape

        # normalize
        query_embd = query_embd / torch.norm(query_embd)
        img_embd = img_embd / torch.norm(img_embd, dim=-1, keepdim=True)

        attn = torch.sum(img_embd * query_embd, dim=-1)
        # attn = -torch.norm(img_embd - query_embd, dim=-1)

        # apply softmax
        attn = attn.reshape((b, -1))
        attn = torch.softmax(attn.float(), dim=-1)
        attn = attn.reshape((b, h, w))
        return attn


def crop_image(img, crop_center_xy, crop_half_size, blur_size=None):
    """
    Args
        img: (h, w, c)
        crop_center_xy: (x, y)
        crop_half_size: (h, w)

    Returns
        img_crop: (h, w, c)
    """
    h, w, c = img.shape
    crop_x_half_size = crop_half_size[0]
    crop_y_half_size = crop_half_size[1]
    crop_center_x, crop_center_y = crop_center_xy
    crop_center_x = int(crop_center_x)
    crop_center_y = int(crop_center_y)
    crop_range_x = np.arange(
        np.maximum(crop_center_x - crop_x_half_size, 0),
        np.minimum(crop_center_x + crop_x_half_size, w),
    )
    crop_range_y = np.arange(
        np.maximum(crop_center_y - crop_y_half_size, 0),
        np.minimum(crop_center_y + crop_y_half_size, h),
    )
    crop_x, crop_y = np.meshgrid(crop_range_x, crop_range_y)
    img_crop = img[crop_y, crop_x, :]
    if blur_size:
        img_crop = cv2.GaussianBlur(img_crop, (blur_size, blur_size), 0)
    pixel_xy_orig_to_crop = lambda xy: (
        xy[0] - crop_range_x[0],
        xy[1] - crop_range_y[0],
    )
    pixel_xy_crop_to_orig = lambda xy: (
        xy[0] + crop_range_x[0],
        xy[1] + crop_range_y[0],
    )
    return img_crop, pixel_xy_orig_to_crop, pixel_xy_crop_to_orig


@torch.no_grad()
def embed_imgs(
    imgs: Union[np.ndarray, List[np.ndarray]],
    preprocess: Callable[[Image.Image], torch.Tensor],
    vit_encoder: Callable[[torch.Tensor], torch.Tensor],
    patch_size,
    device="cuda:0",
    batch_size=8,
    log=False,
):
    """
    Args:
        imgs: (b, h, w, c)
        preprocess: preprocess function takes PIL image and returns (3, h_in, w_in)
        vit_encoder: vit encoder function takes imgs (b, 3, h_in, w_in) and returns (b, h_in * w_in, c) where c is the embedding size
        patch_size: patch size
        device: device
        batch_size: batch size

    Returns:
        embeddings: (b, h_in // patch_size, w_in // patch_size, c)
    """
    
    # preprocessed_images = [
    #     preprocess(Image.fromarray(imgs[i])) for i in range(len(imgs))
    # ]
    # preprocessed_images = torch.stack(preprocessed_images).to(device)
        
    h_in, w_in = preprocess(Image.fromarray(imgs[0])).shape[-2:]

    # Get CLIP embeddings for the images
    embeddings = []
    for i in tqdm(
        range(0, len(imgs), batch_size),
        desc="Extracting CLIP features",
        disable=not log,
    ):
        # batch = preprocessed_images[i : i + batch_size]
        if log:
            print("preprocessing images...")
        batch = [
            preprocess(Image.fromarray(imgs[i])) for i in range(i, min(i + batch_size, len(imgs)))
        ]
        batch = torch.stack(batch)
        if log:
            print("transferring to device", batch.shape, batch.device)
        batch = batch.to(device)
        if log:
            print("embedding...")
        embeddings.append(vit_encoder(batch).detach().cpu())
    embeddings = torch.cat(embeddings, dim=0)

    # Reshape embeddings from flattened patches to patch height and width
    # h_in, w_in = preprocessed_images.shape[-2:]
    # if clip_model_name.startswith("ViT"):
    h_out = h_in // patch_size
    w_out = w_in // patch_size
    # elif clip_model_name.startswith("RN"):
    #     h_out = max(h_in / w_in, 1.0) * model.visual.attnpool.spacial_dim
    #     w_out = max(w_in / h_in, 1.0) * model.visual.attnpool.spacial_dim
    #     h_out, w_out = int(h_out), int(w_out)
    # else:
    #     raise ValueError(f"Unknown CLIP model name: {clip_model_name}")
    # del preprocessed_images
    # torch.cuda.empty_cache()
    embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
    return embeddings


def crop_img_and_match_feature(
    preprocess: Callable[[Image.Image], torch.Tensor],
    vit_encoder: Callable[[torch.Tensor], torch.Tensor],
    input_resolution: int,
    patch_size: int,
    img1: np.ndarray,
    img2: np.ndarray,
    query_xy,
    crop_half_size=None,
    blur_size=None,
    target_xy_prev=None,
    batch_size=8,
    device="cuda:0",
):
    """
    Args:
        preprocess: preprocess function takes PIL image and returns (3, h_in, w_in)
        vit_encoder: vit encoder function takes imgs (b, 3, h_in, w_in) and returns (b, h_in * w_in, c) where c is the embedding size
        input_resolution: input resolution
        patch_size: patch size
        img1: (h, w, c)
        img2: (h, w, c)
        query_xy: (x, y)
        crop_size: (h, w)
        blur_size (int): blur size
        target_xy_prev: (x, y)
        batch_size: batch size
        device: device

    Returns:
        attn: (h, w)
    """

    # assert img1.shape == img2.shape, "img1 and img2 must have the same shape"
    assert len(img1.shape) == 3, "img1 and img2 must have shape (h, w, c)"
    assert len(query_xy) == 2, "query_xy must have shape (x, y)"
    assert (
        crop_half_size is None
        and target_xy_prev is None
        or crop_half_size is not None
        and target_xy_prev is not None
    ), "crop_size and target_xy_prev must be both None or both not None"

    # crop image
    img1_crop = img1
    img2_crop = img2
    pixel_xy_orig_to_crop_1 = None
    pixel_xy_crop_to_orig_2 = None
    if crop_half_size:
        img1_crop, pixel_xy_orig_to_crop_1, pixel_xy_crop_to_orig_1 = crop_image(
            img1, query_xy, crop_half_size, blur_size=blur_size
        )
        img2_crop, pixel_xy_orig_to_crop_2, pixel_xy_crop_to_orig_2 = crop_image(
            img2, target_xy_prev, crop_half_size, blur_size=blur_size
        )

    # convert query_xy to patch coords in img1_crop
    query_x, query_y = query_xy
    if pixel_xy_orig_to_crop_1:
        query_x, query_y = pixel_xy_orig_to_crop_1(query_xy)
    patch_query_x, patch_query_y = scale_coords(
        query_x, query_y, img1_crop.shape, input_resolution
    )
    patch_query_x = patch_query_x // patch_size
    patch_query_y = patch_query_y // patch_size

    # embed cropped
    # NOTE we cannot batch this because we cropped images might have different sizes
    # TODO pad cropped images to the same size
    img1_crop_embd = embed_imgs(
        imgs=img1_crop[None, ...],
        preprocess=preprocess,
        vit_encoder=vit_encoder,
        patch_size=patch_size,
        device=device,
        batch_size=batch_size,
    )[0]
    img2_crop_embd = embed_imgs(
        imgs=img2_crop[None, ...],
        preprocess=preprocess,
        vit_encoder=vit_encoder,
        patch_size=patch_size,
        device=device,
        batch_size=batch_size,
    )[0]

    # query img2_crop_embd
    query_embd = img1_crop_embd[patch_query_y, patch_query_x, :]
    attn = query_embedding(query_embd, img2_crop_embd[None, ...])[0]  # shape (h2, w2)
    attn = torch_utils.to_numpy(attn)

    # find attended path in img2_crop
    # patch_coords_x_2, patch_coords_y_2 = np.meshgrid(np.arange(img2_crop_embd.shape[1]), np.arange(img2_crop_embd.shape[0]))
    # use maximum value of attn as target
    target_y, target_x = np.unravel_index(np.argmax(attn), attn.shape)
    target_x = int(target_x) * patch_size
    target_y = int(target_y) * patch_size
    target_x, target_y = img_coords_resized_to_orig(
        target_x, target_y, img2_crop.shape, input_resolution
    )
    # target_x = np.sum(patch_coords_x_2 * attn) * model.visual.patch_size
    # target_y = np.sum(patch_coords_y_2 * attn) * model.visual.patch_size

    # convert to pixel coords
    if pixel_xy_crop_to_orig_2:
        target_x, target_y = pixel_xy_crop_to_orig_2((target_x, target_y))

    return (target_x, target_y), attn, img1_crop, img2_crop


def iterative_feature_matching(
    preprocess: Callable[[Image.Image], torch.Tensor],
    vit_encoder: Callable[[torch.Tensor], torch.Tensor],
    input_resolution: int,
    patch_size: int,
    img1: np.ndarray,
    img2: np.ndarray,
    query_xy,
    crop_half_sizes=[40, 25, 20],
    blur_sizes=[5, 11, 13],
    batch_size=8,
    device="cuda:0",
    log=True,
):
    """
    Args:
        preprocess: preprocess function takes PIL image and returns (3, h_in, w_in)
        vit_encoder: vit encoder function takes imgs (b, 3, h_in, w_in) and returns (b, h_in * w_in, c) where c is the embedding size
        input_resolution: input resolution
        patch_size: patch size
        img1: (h, w, c)
        img2: (h, w, c)
        query_xy: (x, y)
        crop_half_sizes: list of crop half sizes
        blur_sizes: list of blur sizes. NOTE blur size must be odd
        batch_size: batch size
        device: device

    Returns:
        target_xy: (x, y)
        attns: list of attns
        img1_crops: list of img1_crops
        img2_crops: list of img2_crops
    """
    attns = []
    img1_crops = []
    img2_crops = []
    crop_half_size = None
    blur_size = None
    target_xy = None
    for i in range(len(crop_half_sizes) + 1):
        target_xy, attn, img1_crop, img2_crop = crop_img_and_match_feature(
            preprocess=preprocess,
            vit_encoder=vit_encoder,
            input_resolution=input_resolution,
            patch_size=patch_size,
            img1=img1,
            img2=img2,
            query_xy=query_xy,
            crop_half_size=(crop_half_size, crop_half_size) if crop_half_size else None,
            blur_size=blur_size,
            target_xy_prev=target_xy,
            batch_size=batch_size,
            device=device,
        )

        attns.append(attn)
        img1_crops.append(img1_crop)
        img2_crops.append(img2_crop)

        if i < len(crop_half_sizes):
            crop_half_size = crop_half_sizes[i]
            blur_size = blur_sizes[i]
        
        if log:
            print(f"iter {i}: target_xy={target_xy}")

    return target_xy, attns, img1_crops, img2_crops


def create_img_crop_origins_grid(img_shape, crop_half_size: int) -> np.ndarray:
    """
    Args:
        img_shape: (h, w, 3)
        crop_size: crop size
        blur_size: blur size

    Returns:
        patch_origins_xy: (n, 2)
    """
    h, w = img_shape[:2]
    crop_size = 2 * crop_half_size
    n_h = h // crop_size
    n_w = w // crop_size

    patch_origins_xy = []

    for i in range(n_h):
        for j in range(n_w):
            patch_origin_xy = (
                j * crop_size,
                i * crop_size,
            )
            patch_origins_xy.append(patch_origin_xy)
            # add another patch which overlaps with the previous patch by half
            if (patch_origin_xy[0] + crop_size + crop_half_size) <= w and (
                patch_origin_xy[1] + crop_size + crop_half_size
            ) <= h:
                patch_origin_xy = (
                    j * crop_size + crop_half_size,
                    i * crop_size + crop_half_size,
                )
                patch_origins_xy.append(patch_origin_xy)
    return np.array(patch_origins_xy)


def create_img_crop_origins(
    img_shape, crop_centers_xy, crop_half_size: int
) -> np.ndarray:
    """
    Args:
        img_shape: (h, w, 3)
        crop_centers_xy: (n, 2)
        crop_half_size: crop half size
    """
    h, w = img_shape[:2]
    crop_size = 2 * crop_half_size
    crop_origins = np.zeros_like(crop_centers_xy)

    # makes sure crop centers are at least crop_half_size away from the image border
    crop_origins[:, 0] = np.clip(crop_centers_xy[:, 0], 0, w - crop_size)
    crop_origins[:, 1] = np.clip(crop_centers_xy[:, 1], 0, h - crop_size)

    return crop_origins


def extract_img_crops(
    img: np.ndarray,
    patch_origins_xy: Union[List[Tuple[int, int]], np.ndarray],
    crop_half_size: int,
    blur_size: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Args:
        img: (h, w, 3)
        patch_origins: list of patch origins or (n, 2)
    """
    img_patches = []
    for i in range(len(patch_origins_xy)):
        patch_origin_xy = patch_origins_xy[i]
        img_crop, _, _ = crop_image(
            img,
            crop_center_xy=(
                patch_origin_xy[0] + crop_half_size,
                patch_origin_xy[1] + crop_half_size,
            ),
            crop_half_size=(crop_half_size, crop_half_size),
            blur_size=blur_size,
        )
        img_patches.append(img_crop)
    return img_patches


def multi_query_embedding(query_embd: torch.Tensor, img_embd: torch.Tensor):
    """
    Args:
        query_embd: (nq, c)
        img_embd: (nq, h, w, c)
    Returns:
        attn: (nq, h, w)
    """
    with torch.no_grad():
        nq, h, w, c = img_embd.shape

        # normalize
        query_embd = query_embd / torch.norm(
            query_embd, dim=-1, keepdim=True
        )  # (nq, c)
        img_embd = img_embd / torch.norm(img_embd, dim=-1, keepdim=True)  # (b, h, w, c)

        attn = torch.einsum(
            "nc,nhwc->nhw", query_embd.float(), img_embd.float()
        )  # (nq, h, w)

        # apply softmax
        attn = attn.reshape((nq, -1))
        attn = torch.softmax(attn, dim=-1)
        attn = attn.reshape((nq, h, w))
        # attn = attn.reshape((b, -1))
        # attn = torch.softmax(attn, dim=-1)
        # attn = attn.reshape((b, h, w))
        return attn


def iterative_feature_matching_fast(
    preprocess: Callable[[Image.Image], torch.Tensor],
    vit_encoder: Callable[[torch.Tensor], torch.Tensor],
    input_resolution: int,
    patch_size: int,
    img1: np.ndarray,
    img2: np.ndarray,
    queries_xy,
    crop_half_sizes=[40, 25, 20],
    blur_sizes=[5, 11, 13],
    batch_size=8,
    device="cuda:0",
    log=True,
):
    """
    Args:
        preprocess: preprocess function takes PIL image and returns (3, h_in, w_in)
        vit_encoder: vit encoder function takes imgs (b, 3, h_in, w_in) and returns (b, h_in * w_in, c) where c is the embedding size
        input_resolution: input resolution
        patch_size: patch size
        img1: (h, w, c)
        img2: (h, w, c)
        queries_xy: (nq, 2)
        crop_sizes: list of crop sizes
        blur_sizes: list of blur sizes. NOTE blur size must be odd
        batch_size: batch size
        device: device

    Returns:
        target_xy: (nq, x, y)
    """
    nq = queries_xy.shape[0]
    target_xy: Union[np.ndarray, None] = None

    all_img_crops = []
    all_attns = []
    all_query_img_crop_ids = []
    all_target_img_crop_ids = []

    cur_img1_crops = [img1]
    cur_img2_crops = [img2]
    cur_img1_crop_origins_xy = np.zeros((nq, 2), dtype=int)
    cur_img2_crop_origins_xy = np.zeros((nq, 2), dtype=int)
    cur_crop_half_size = 0
    for i in range(len(crop_half_sizes) + 1):
        all_img_crops.append((cur_img1_crops, cur_img2_crops))

        # embed img1 and img2 crops
        cur_img_crops = cur_img1_crops + cur_img2_crops
        cur_img_crop_embeddings: torch.Tensor = embed_imgs(
            imgs=cur_img_crops,
            preprocess=preprocess,
            vit_encoder=vit_encoder,
            patch_size=patch_size,
            device=device,
            batch_size=batch_size,
            log=log,
        )
        cur_img1_crop_embd = cur_img_crop_embeddings[
            : len(cur_img1_crops)
        ]  # shape (n1, hp, wp, c)
        cur_img2_crop_embd = cur_img_crop_embeddings[
            len(cur_img1_crops) :
        ]  # shape (n2, hp, wp, c)

        # for eqach query keypoint select closest crop in img1
        cur_img1_crop_centers_xy = cur_img1_crop_origins_xy + cur_crop_half_size
        query_img1_crop_ids = np.argmin(
            np.linalg.norm(
                queries_xy[:, None, :] - cur_img1_crop_centers_xy[None, :, :], axis=-1
            ),
            axis=-1,
        )  # shape (nq)
        all_query_img_crop_ids.append(query_img1_crop_ids)
        query_img1_crop_origins_xy = cur_img1_crop_origins_xy[query_img1_crop_ids]
        query_img1_crop_embd = cur_img1_crop_embd[
            query_img1_crop_ids
        ]  # shape (nq, hp, wp, c)

        # for each target keypoint select closest crop in img2
        cur_img2_crop_centers_xy = cur_img2_crop_origins_xy + cur_crop_half_size
        if target_xy is None:
            target_img2_crop_ids: np.ndarray = np.zeros((nq,), dtype=int)
        else:
            target_img2_crop_ids = np.argmin(
                np.linalg.norm(
                    target_xy[:, None, :] - cur_img2_crop_centers_xy[None, :, :],
                    axis=-1,
                ),
                axis=-1,
            ).astype(
                int
            )  # shape (nq)
        all_target_img_crop_ids.append(target_img2_crop_ids)
        target_img2_crop_origins_xy = cur_img2_crop_origins_xy[target_img2_crop_ids]
        target_img2_crop_embd = cur_img2_crop_embd[
            target_img2_crop_ids
        ]  # shape (nq, hp, wp, c)

        # for each query keypoint select closest patch in corresponding crop in img1
        patch_query_xy = queries_xy - query_img1_crop_origins_xy
        patch_query_x, patch_query_y = img_coords_orig_to_resized_np(
            patch_query_xy[:, 0],
            patch_query_xy[:, 1],
            cur_img1_crops[0].shape,
            input_resolution,
        )
        patch_query_xy = np.stack([patch_query_x, patch_query_y], axis=-1)
        query_img1_crop_patch_xy = np.floor(patch_query_xy / patch_size).astype(int)
        # print("query_img1_crop_patch_xy", query_img1_crop_patch_xy)
        query_img1_crop_embd_of_patch = query_img1_crop_embd[
            np.arange(nq),
            query_img1_crop_patch_xy[:, 1],
            query_img1_crop_patch_xy[:, 0],
        ]  # shape (nq, c)

        # compute attention between each query point and its corresponding crop in img2
        attn = multi_query_embedding(
            query_img1_crop_embd_of_patch, target_img2_crop_embd
        )  # shape (nq, hp, wp)
        attn = tF.gaussian_blur(attn, kernel_size=[3,3], sigma=[0.5, 0.5])  # , sigma=[1, 1]

        attn_np: np.ndarray = torch_utils.to_numpy(attn)
        # attn_np = cv2.GaussianBlur(attn_np, (3, 3), 0)
        
        all_attns.append(attn_np)

        hp, wp = attn_np.shape[1:]
        attn_np = attn_np.reshape((nq, hp * wp))  # shape (nq, hp * wp)
        attn_argmax = np.argmax(attn_np, axis=-1)  # shape (nq,)
        target_y, target_x = np.unravel_index(
            attn_argmax, (hp, wp)
        )  # shape (nq,), (nq,)
        target_x *= patch_size
        target_y *= patch_size

        target_x, target_y = img_coords_resized_to_orig_np(
            target_x, target_y, cur_img2_crops[0].shape, input_resolution
        )

        target_x = target_img2_crop_origins_xy[:, 0] + target_x
        target_y = target_img2_crop_origins_xy[:, 1] + target_y
        target_xy = np.stack([target_x, target_y], axis=-1)  # shape (nq, 2)

        if i < len(crop_half_sizes):
            cur_crop_half_size = crop_half_sizes[i]
            blur_size = blur_sizes[i]
            # cur_img1_crop_origins = create_img_crop_origins(img1.shape, queries_xy, cur_crop_half_size)
            # cur_img2_crop_origins = create_img_crop_origins(img2.shape, (target_x, target_y), cur_crop_half_size)
            cur_img1_crop_origins_xy = create_img_crop_origins_grid(
                img1.shape, cur_crop_half_size
            )
            cur_img2_crop_origins_xy = create_img_crop_origins_grid(
                img2.shape, cur_crop_half_size
            )
            cur_img1_crop_centers_xy = cur_img1_crop_origins_xy + cur_crop_half_size
            cur_img2_crop_centers_xy = cur_img2_crop_origins_xy + cur_crop_half_size
            query_img1_crop_ids = np.argmin(
                np.linalg.norm(
                    queries_xy[:, None, :] - cur_img1_crop_centers_xy[None, :, :],
                    axis=-1,
                ),
                axis=-1,
            ).astype(
                int
            )  # shape (nq)
            target_img2_crop_ids = np.argmin(
                np.linalg.norm(
                    target_xy[:, None, :] - cur_img2_crop_centers_xy[None, :, :],
                    axis=-1,
                ),
                axis=-1,
            ).astype(
                int
            )  # shape (nq)
            query_img1_crop_ids_set = list(set(query_img1_crop_ids))
            target_img2_crop_ids_set = list(set(target_img2_crop_ids))

            # filter crops
            cur_img1_crop_origins_xy = cur_img1_crop_origins_xy[query_img1_crop_ids_set]
            cur_img2_crop_origins_xy = cur_img2_crop_origins_xy[
                target_img2_crop_ids_set
            ]

            cur_img1_crop_centers_xy = cur_img1_crop_centers_xy[query_img1_crop_ids_set]
            cur_img2_crop_centers_xy = cur_img2_crop_centers_xy[
                target_img2_crop_ids_set
            ]

            cur_img1_crops = extract_img_crops(
                img1, cur_img1_crop_origins_xy, cur_crop_half_size, blur_size
            )
            cur_img2_crops = extract_img_crops(
                img2, cur_img2_crop_origins_xy, cur_crop_half_size, blur_size
            )

        # TODO replace with log function from library
        if log:
            print(f"iter {i}")

    return target_xy, all_attns, all_img_crops, all_query_img_crop_ids, all_target_img_crop_ids
