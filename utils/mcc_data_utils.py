import numpy as np
import torch
import cv2


def make_square_image(im, pad_value):
    if im.shape[0] > im.shape[1]:
        diff = im.shape[0] - im.shape[1]
        return torch.cat([im, (torch.zeros((im.shape[0], diff, im.shape[2])) + pad_value)], dim=1)
    else:
        diff = im.shape[1] - im.shape[0]
        return torch.cat([im, (torch.zeros((diff, im.shape[1], im.shape[2])) + pad_value)], dim=0)


def normalize_xyz(seen_xyz):
    seen_xyz = seen_xyz / (seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].var(dim=0) ** 0.5).mean()
    seen_xyz = seen_xyz - seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
    return seen_xyz


def normalize_xyz_basic(xyz, mean, max_dist):
    xyz = xyz - mean
    xyz /= max_dist
    return xyz

def unnormalize_xyz_basic(xyz, mean, max_dist):
    xyz = np.copy(xyz)
    xyz *= max_dist
    xyz += mean
    return xyz


def normalize_xyz_v2(seen_xyz):
    is_valid = torch.isfinite(seen_xyz.sum(dim=-1))
    seen_xyz_mean = seen_xyz[is_valid].mean(dim=0)
    seen_xyz = seen_xyz - seen_xyz_mean
    seen_xyz_max_dist = torch.max(torch.abs(seen_xyz[is_valid]))
    seen_xyz /= seen_xyz_max_dist
    return seen_xyz, seen_xyz_mean, seen_xyz_max_dist


def compute_masked_mean_and_max_dist(img: np.ndarray, xyz: np.ndarray, seg: np.ndarray):
    """
    Args:
        img: (H, W, 3)
        xyz: (H*W, 3)
        seg: (H, W)
    """
    depth_pts = xyz.reshape(img.shape)
    img_height, img_width = img.shape[:2]
    mask = cv2.resize(seg.astype(float), (img_width, img_height)).astype(bool) # NOTE width first due to cv2
    depth_pts[~mask] = float('inf')

    masked_mean = depth_pts[mask].mean(axis=0)
    masked_max_dist = np.max(np.abs(depth_pts[mask]) - masked_mean)

    return masked_mean, masked_max_dist


def get_grid(B, device, co3d_world_size, granularity):
    """
    Description
    -----------
    Construct a grid of points in the 3D world. The grid is centered at the origin and has a total size of co3d_world_size.
    """
    N = int(np.ceil(co3d_world_size / granularity))
    grid_unseen_xyz = torch.zeros((N, N, N, 3), device=device)
    for i in range(N):
        grid_unseen_xyz[i, :, :, 0] = i
    for j in range(N):
        grid_unseen_xyz[:, j, :, 1] = j
    for k in range(N):
        grid_unseen_xyz[:, :, k, 2] = k
    grid_unseen_xyz /= N
    grid_unseen_xyz -= 0.5
    grid_unseen_xyz *= co3d_world_size
    # grid_unseen_xyz /= (N / 2.0)
    grid_unseen_xyz = grid_unseen_xyz.reshape((1, -1, 3)).repeat(B, 1, 1)
    return grid_unseen_xyz



def sample_uniform_semisphere(B, N, semisphere_size, device):
    for _ in range(100):
        points = torch.empty(B * N * 3, 3, device=device).uniform_(-semisphere_size, semisphere_size)
        points[..., 2] = points[..., 2].abs()
        dist = (points ** 2.0).sum(axis=-1) ** 0.5
        if (dist < semisphere_size).sum() >= B * N:
            return points[dist < semisphere_size][:B * N].reshape((B, N, 3))
        else:
            print('resampling sphere')


def get_grid_semisphere(B, granularity, semisphere_size, device):
    n_grid_pts = int(semisphere_size / granularity) * 2 + 1
    grid_unseen_xyz = torch.zeros((n_grid_pts, n_grid_pts, n_grid_pts // 2 + 1, 3), device=device)
    for i in range(n_grid_pts):
        grid_unseen_xyz[i, :, :, 0] = i
        grid_unseen_xyz[:, i, :, 1] = i
    for i in range(n_grid_pts // 2 + 1):
        grid_unseen_xyz[:, :, i, 2] = i
    grid_unseen_xyz[..., :2] -= (n_grid_pts // 2.0)
    grid_unseen_xyz *= granularity
    dist = (grid_unseen_xyz ** 2.0).sum(axis=-1) ** 0.5
    grid_unseen_xyz = grid_unseen_xyz[dist <= semisphere_size]
    return grid_unseen_xyz[None].repeat(B, 1, 1)


# TODO extremely slow
def get_min_dist(a, b, slice_size=1000):
    all_min, all_idx = [], []
    for i in range(int(np.ceil(a.shape[1] / slice_size))):
        start = slice_size * i
        end   = slice_size * (i + 1)
        # B, n_queries, n_gt
        dist = ((a[:, start:end] - b) ** 2.0).sum(axis=-1) ** 0.5
        # B, n_queries
        cur_min, cur_idx = dist.min(axis=2)
        all_min.append(cur_min)
        all_idx.append(cur_idx)
    return torch.cat(all_min, dim=1), torch.cat(all_idx, dim=1)


def construct_uniform_semisphere(gt_xyz, gt_rgb, semisphere_size, n_queries, dist_threshold, is_train, granularity):
    B = gt_xyz.shape[0]
    device = gt_xyz.device
    if is_train:
        unseen_xyz = sample_uniform_semisphere(B, n_queries, semisphere_size, device)
    else:
        unseen_xyz = get_grid_semisphere(B, granularity, semisphere_size, device)
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    labels = dist < dist_threshold
    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[labels] = torch.gather(gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3))[labels]
    return unseen_xyz, unseen_rgb, labels.float()


def construct_uniform_grid(gt_xyz, gt_rgb, co3d_world_size, n_queries, dist_threshold, is_train, granularity):
    """
    Args:
        gt_xyz: (B, N, 3)
        gt_rgb: (B, N, 3)
        co3d_world_size: float
        n_queries: int
        dist_threshold: float
        is_train: bool
        granularity: float
    """
    B = gt_xyz.shape[0]
    device = gt_xyz.device
    if is_train:
        unseen_xyz = torch.empty((B, n_queries, 3), device=device).uniform_(-co3d_world_size, co3d_world_size)
    else:
        unseen_xyz = get_grid(B, device, co3d_world_size, granularity)  # (B, n_queries, 3)
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    is_seen = dist < dist_threshold
    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[is_seen] = torch.gather(gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3))[is_seen]
    return unseen_xyz, unseen_rgb, is_seen.float()


def remove_inf_from_xyz(xyz, replace_val=-100):
    xyz_valid = xyz.clone().detach()
    is_valid = torch.isfinite(xyz_valid.sum(axis=-1))
    xyz_valid[~is_valid] = replace_val
    return xyz_valid, is_valid


def sample_unseen_xyz(
    batch_size, 
    grid_size=1,
    grid_granularity=0.01,
    device="cuda"
):
    unseen_xyz = get_grid(batch_size, device, grid_size, grid_granularity)
    return unseen_xyz


def get_model_input(img, depth_pts, seg, grid_size=1, grid_granularity=0.01, device="cuda"):
    """
    Args:
        img: (3, W, H) torch.Tensor
        depth_pts: (H, W, 3) torch.Tensor
        seg: (H, W) torch.Tensor
        grid_size: float
        grid_granularity: float
        device: str
    """

    # normalize rgb image
    seen_rgb = (torch.tensor(img).float() / 255)[..., [2, 1, 0]] # shape: (H, W, 3)
    img_height, img_width = seen_rgb.shape[:2]

    # TODO why interpolate to the same size?
    #       => This should be depth_pts.shape[:2] ???
    depth_pts = torch.tensor(depth_pts).float()  # shape: (H, W, 3)
    if depth_pts.shape[:2] != (img_height, img_width):
        depth_pts = torch.nn.functional.interpolate(
            depth_pts.permute(2, 0, 1)[None],  # shape: (1, 3, H, W)
            size=[img_height, img_width],
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)  # shape: (H, W, 3)

    seen_xyz = depth_pts
    
    mask = torch.tensor(cv2.resize(seg.astype(float), (img_width, img_height))).bool()
    seen_xyz[~mask] = float('inf')

    # seen_xyz = mcc_data_utils.normalize_xyz(seen_xyz)
    seen_xyz, _, _ = normalize_xyz_v2(seen_xyz)

    # cut out masked region
    bottom, right = mask.nonzero().max(dim=0)[0]
    top, left = mask.nonzero().min(dim=0)[0]
    bottom = bottom + 40
    right = right + 40
    top = max(top - 40, 0)
    left = max(left - 40, 0)
    seen_xyz = seen_xyz[top:bottom+1, left:right+1]
    seen_rgb = seen_rgb[top:bottom+1, left:right+1]
    seen_xyz = make_square_image(seen_xyz, float('inf'))
    seen_rgb = make_square_image(seen_rgb, 0)

    # resize rgb image to 800x800
    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],  # shape: (1, 3, H, W)
        size=[800, 800],
        mode="bilinear",
        align_corners=False,
    )  # shape: (1, 3, 800, 800)

    # resize depth points image to 112x112
    seen_xyz = torch.nn.functional.interpolate(
        seen_xyz.permute(2, 0, 1)[None],  # shape: (1, 3, H, W)
        size=[112, 112],
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)  # shape: (1, 112, 112, 3)

    # sample query points and create dummy rgb, is_valid, and is_seen tensors
    valid_seen_xyz, is_valid = remove_inf_from_xyz(seen_xyz)
    unseen_xyz = sample_unseen_xyz(
        batch_size = 1,
        grid_size=grid_size,
        grid_granularity=grid_granularity,
        device=device
    )
    is_seen = torch.zeros((unseen_xyz.shape[0], unseen_xyz.shape[1]), device=device).bool()
    unseen_rgb = torch.zeros_like(unseen_xyz)

    return {
        "seen_xyz": valid_seen_xyz,
        "seen_rgb": seen_rgb,
        "unseen_xyz": unseen_xyz,  # query points
        "unseen_rgb": unseen_rgb,  # only used for loss => not needed for inference
        "is_seen": is_seen,  # only used for loss => not needed for inference
        "is_valid": is_valid,  # only used for loss => not needed for inference
    }