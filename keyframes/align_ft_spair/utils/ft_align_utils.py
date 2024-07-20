from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft
from skimage.feature import peak_local_max
from keyframes.align_ft_spair.utils import geom_utils, img_utils


def rot(x, y, s, angle_deg):
    angle_rad = np.radians(angle_deg)
    x_c = x - s // 2
    y_c = y - s // 2
    x_new = x_c * np.cos(angle_rad) - y_c * np.sin(angle_rad)
    y_new = x_c * np.sin(angle_rad) + y_c * np.cos(angle_rad)
    x_new += s // 2
    y_new += s // 2
    return x_new, y_new


def transform_point(x, y, s, flip=0, angle_deg=0):
    if flip == 1:
        x = s - 1 - x
    if angle_deg:
        x, y = rot(x, y, s, -angle_deg)
    return x, y


def transform_point_inv(x, y, s, flip=0, angle_deg=0):
    if angle_deg:
        x, y = rot(x, y, s, angle_deg)
    if flip == 1:
        x = s - x
    return x, y

def parse_filename(filename):
    """
    Args:
        filename (str): filename of the form "{obj_name}_flip={flip}_rot={angle_degree}.{ext}"
    """
    filename = filename.split(".")[0]
    filename_split = filename.split("_")
    flip = int(filename_split[-2].split("=")[1])
    angle_degree = int(filename_split[-1].split("=")[1])
    obj_name = "_".join(filename_split[:-2])
    return obj_name, flip, angle_degree

def load_img_embds(img_names, rot_offsets, root_dir, angle_deg=0, flip=0):
    data_folder = f'{root_dir}/data/img_transformed'
    img_embds = []
    for i in range(len(img_names)):
        img_embd_filename = f'{img_names[i]}_flip={flip}_rot={(angle_deg + rot_offsets[i]) % 360}.pt'
        img_embd = torch.load(os.path.join(data_folder, img_embd_filename))
        img_embds.append(img_embd)
    img_embds = torch.concatenate(img_embds, dim=0) # N, C, H, W
    return img_embds

def img_to_embedding_coords(x, y, img_w=768, embd_w=48, img_h=None, embd_h=None):
    """
    Args:
        x (int): x coordinate in image
        y (int): y coordinate in image
        img_w (int): width of image
        embd_w (int): width of embedding
        img_h (int): height of image
        embd_h (int): height of embedding
    """
    embd_h = embd_h or embd_w
    img_h = img_h or img_w
    x = x * embd_w // img_w
    y = y * embd_h // img_h
    return int(x), int(y)

def embd_to_img_coords(x_embd, y_embd, img_w=768, embd_w=48, img_h=None, embd_h=None):
    """
    Args:
        x_embd (int): x coordinate in embedding
        y_embd (int): y coordinate in embedding
        img_w (int): width of image
        embd_w (int): width of embedding
        img_h (int): height of image
        embd_h (int): height of embedding
    """
    embd_h = embd_h or embd_w
    img_h = img_h or img_w
    x = x_embd * img_w // embd_w
    y = y_embd * img_h // embd_h
    return int(x), int(y)


def find_correspondences(query_embd, target_embds):
    """
    Args:
        query_embd: (1,C)
        target_embds: (N, C, H, W)
    """
    # compute feature similarities
    n_q, c_q = query_embd.size()
    assert n_q == 1, "query must be a single feature vector"
    n, c, h, w = target_embds.size()
    assert c == c_q, "query and target feature dimensions must match"

    # normalize
    query_embd = F.normalize(query_embd, dim=1) # (1, C)
    target_embds = F.normalize(target_embds, dim=1) # (N, C, H, W)

    target_embds = target_embds.view(n, c, -1) # N, C, HW
    cos_map = torch.matmul(query_embd, target_embds) # (1, C) x (N, C, HW) = (N, HW)
    target_embds = target_embds.view(n, c, h, w) # N, C, H, W
    cos_map = cos_map.view(n, h, w).cpu().numpy()

    # extract feature with maximum cosinues similarity
    most_similar_features = []
    all_max_yx = []
    for i in range(n):
        max_yx = np.unravel_index(cos_map[i].argmax(), cos_map[i].shape)
        all_max_yx.append(max_yx)
        most_similar_ft = target_embds[i, :, max_yx[0], max_yx[1]].view(1, -1)  # (1, C)
        most_similar_features.append(most_similar_ft)
    most_similar_features = torch.concatenate(most_similar_features, dim=0)

    return most_similar_features, all_max_yx


def get_img_embd_at_xy_orig(x, y, embd, img_w, embd_w, angle_deg, flip):
    """
    Args:
        x (int): x coordinate in original (non transformed) image
        y (int): y coordinate in original (non transformed) image
        embd (Tensor): (C, H, W) of the transformed image
        img_w (int): width of image
        embd_w (int): width of embedding
        angle_deg (int): angle of rotation
        flip (int): flip
    Returns:
        ft (Tensor): (1, C)
    """
    x_img_rot, y_img_rot = transform_point(x, y, img_w, flip, angle_deg)
    x_embd_rot, y_embd_rot = img_to_embedding_coords(x_img_rot, y_img_rot, img_w, embd_w)
    if x_embd_rot >= 0 and x_embd_rot < embd_w and y_embd_rot >= 0 and y_embd_rot < embd_w:
        ft = embd[:, y_embd_rot, x_embd_rot].view(1, -1)  # (1, C)
        return ft, x_embd_rot, y_embd_rot
    else:
        return None, x_embd_rot, y_embd_rot


def feature_x_rotated_feature(
    query_embd: torch.Tensor, 
    img_names: List[str], 
    rot_offsets: List[int],
    root_dir: str, 
    img_w: int = 768,
    embd_w: int = 48
):
    """ 1) For each image, find most similar points to query_img in non-rotated case. 
        2) Then, for each img, point, and angle, select img feature at rotate(pt, angle)
    """
    
    # find correspondences in non-rotated case
    img_embds = load_img_embds(img_names, rot_offsets, root_dir, 0, 0) # N, C, H, W
    _, all_max_yx = find_correspondences(query_embd, img_embds)

    # get features at rotated coords
    features = []
    for angle_deg in tqdm(list(np.arange(0,360,5))):
        img_embds = load_img_embds(img_names, rot_offsets, root_dir, angle_deg, 0) # N, C, H, W
        features_deg = []
        for i in range(len(all_max_yx)):
            x_img, y_img = embd_to_img_coords(all_max_yx[i][1], all_max_yx[i][0], img_w, embd_w)
            ft,_,_ = get_img_embd_at_xy_orig(x_img, y_img, img_embds[i], img_w, embd_w, angle_deg, 0)
            features_deg.append(ft)
        features_deg = torch.concatenate(features_deg, dim=0)
        features.append(features_deg)

    # turn list to tensor
    features = torch.stack(features, dim=0) # N, C
    return features


def rotated_feature_x_rotated_feature(
    x_ref_img: int, 
    y_ref_img: int, 
    img_w: int, 
    embd_w: int, 
    img_names: List[str], 
    rot_offsets: List[int], 
    root_dir: str
):
    """ For each angle, select query feature at rotate((x_ref_img, y_ref_img), angle) 
        and return the most similar feature to the query for all images
    """
    
    query_features = []
    retrieved_features = []
    for angle_deg in tqdm(list(np.arange(0,360,5))):
        img_embds = load_img_embds(img_names, rot_offsets, root_dir, angle_deg, 0) # N, C, H, W

        # normalize features
        img_embds = F.normalize(img_embds, dim=1) # (N+1, C, H, W)

        # get ref embd
        x_rot_ref, y_rot_ref = transform_point(x_ref_img, y_ref_img, img_w, flip=0, angle_deg=angle_deg)
        x_ref_embd, y_ref_embd = img_to_embedding_coords(x_rot_ref, y_rot_ref, img_w, embd_w)
        ref_embd_vec = img_embds[0, :, y_ref_embd, x_ref_embd].view(1, -1)  # 1, C
        query_features.append(ref_embd_vec)

        # compute feature similarities
        target_embds = img_embds[1:] # N, C, H, W
        target_embds = target_embds.view(target_embds.size(0), target_embds.size(1), -1) # N, C, HW
        cos_map = torch.matmul(ref_embd_vec, target_embds) # (1, C) x (N, C, HW) = (N, HW)
        cos_map = cos_map.view(cos_map.size(0), img_embds.size(2), img_embds.size(3)).cpu().numpy()

        # extract feature with maximum cosinues similarity
        most_similar_features= []
        for i in range(1, len(img_names)):
            max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)
            most_similar_ft = img_embds[i, :, max_yx[0], max_yx[1]].view(1, -1)  # (1, C)
            most_similar_features.append(most_similar_ft)
        retrieved_features.append(most_similar_features)

    # turn list to tensor
    query_features = torch.concatenate(query_features, dim=0)
    for i in range(len(retrieved_features)):
        retrieved_features[i] = torch.concatenate(retrieved_features[i], dim=0)
    retrieved_features = torch.stack(retrieved_features)
    retrieved_features.shape
    return query_features, retrieved_features


def plot_similar_features(query_features, most_similar_features, repeats=20):
    """
    Args:
        - query_features: (#angles, C)
        - most_similar_features: (#angles, #t-shirts, C)
    Usage:
        # plot
        px.imshow(attn_repeat.cpu().numpy())
    """
    attn = query_features[:, None, :] * most_similar_features # (#angles, #t-shirts, C)

    # TODO why do we take norm here?
    # => maybe for fitting cosinus similarity

    # attn = F.normalize(attn, dim=-1) # (#angles, #t-shirts, C)

    # average does not change norm
    attn = torch.sum(attn, dim=1, keepdim=False) / attn.shape[1] # (#angles, C)
    # attn = F.normalize(attn, dim=-1) # (#angles, C)

    # make image
    attn_repeat = torch.repeat_interleave(attn, repeats, dim=0)

    return attn_repeat


# ==============
# ft_aling
# ==============


def fit_line(x, k=4):
    x_fft = fft(x)
    x_fft_abs = np.abs(x_fft)
    x_fft_abs_sorted = np.sort(x_fft_abs)
    bound = x_fft_abs_sorted[-k]
    x_fft[x_fft_abs < bound] = 0
    x_fit = ifft(x_fft)
    return np.real(x_fit)


def sample_curves(ft_attn: torch.Tensor, ft_angles: np.ndarray, sample_angles: np.ndarray, selected_features: List[int]):
    """ Sample n=#selected_features curves at sample_angles
    Args:
        ft_attn: (#angles, C)
        selected_features: (n)
        ft_angles: (#angles)
        sample_angles: (#samples)
    Returns:
        sample_ft_attn: (n, #samples)
    """
    sample_ft_attn = torch.empty((len(selected_features), len(sample_angles)))
    for i, selected_feature in enumerate(selected_features):
        y = ft_attn[:, selected_feature].cpu().numpy()
        y = fit_line(y, k=15)
        f = interp1d(ft_angles, y, kind='cubic')
        sample_ft_attn[i] = torch.tensor(np.array(f(sample_angles)))
    return sample_ft_attn


def compute_distance(query_ft: torch.Tensor, test_ft: torch.Tensor, selected_features: List[int], curve_samples: torch.Tensor):
    """ Compute distance between query_ft and test_ft
    Args:
        query_ft (torch.Tensor): shape (D1)
        test_ft (torch.Tensor): shape (D1)
        selected_features (List[int]): shape (D2)
        curve_samples (torch.Tensor): shape (D2, n_samples)
    Returns:
        distance (torch.Tensor): shape (1)
    """
    # normalize
    # query_ft = F.normalize(query_ft, dim=0)
    # test_ft = F.normalize(test_ft, dim=0)

    # attention per feature item
    attn = query_ft * test_ft
    attn = attn[selected_features]

    # compute distance to curve_samples and return minimum
    # distances = torch.sum((curve_samples - attn[:, None])**2, dim=0)
    distances = torch.sum(torch.abs(curve_samples - attn[:, None]), dim=0)
    return torch.min(distances)

def compute_attn(query: torch.Tensor, test: torch.Tensor, enable_normalize=True, enable_softmax=True):
    """ Compute attention between query and test tensors.
    Args:
        kpt_embds: tensor of shape (n or 1, C)
        test_embd: tensor of shape (n or 1, C, h, w)
    Returns:
        attn: tensor of shape (n, h, w)
    """
    _, _, h, w = test.shape
    if enable_normalize:
        query = F.normalize(query, dim=1)
        test = F.normalize(test, dim=1)
    attn = query[:,:,None,None] * test  # (n_samples, C, h, w)
    attn = torch.sum(attn, dim=1)
    if enable_softmax:
        attn = attn.view(-1, h*w)
        attn = F.softmax(attn, dim=1)
        attn = attn.view(-1, h, w)
    return attn

def find_kpt_index(query_embd, train_embds, kpt_embd_coords, enable_left_right_check=True):
    """
    Args:
        query_embd: tensor of shape (1, C)
        train_embds: tensor of shape (n_samples, C, h, w)
        kpt_embd_coords: list containing n_samples tensors of shape (n_kpts, 3) where the 3rd dim is the kpt index
    """
    attn = compute_attn(query_embd, train_embds)  # (n_samples, h, w)

    # extract attns at keypoint coords
    n_kpts, n_samples = 30, len(kpt_embd_coords)
    all_kpt_attn = torch.zeros((n_samples, n_kpts))
    for i in range(n_samples):
        if kpt_embd_coords[0].size(0) > 0: 
            x_coords = kpt_embd_coords[i][:,0]
            y_coords = kpt_embd_coords[i][:,1]
            kpt_indices = kpt_embd_coords[i][:,2]
            kpt_attns = attn[i,y_coords,x_coords]
            all_kpt_attn[i,kpt_indices] = kpt_attns.cpu()
    best_sample_kpt_attn = torch.max(all_kpt_attn, dim=0).values
    kpt_index_tensor = torch.argmax(best_sample_kpt_attn)
    kpt_index = kpt_index_tensor.item()

    
    # if kpt_index has left/right issue, 
    # only consider images where both instances are present
    sample_selection = []
    if enable_left_right_check and kpt_index >= 4 and kpt_index <= 21:
        kpt_index_1 = kpt_index - (kpt_index % 2)
        kpt_index_2 = kpt_index_1 + 1
        for i in range(n_samples):
            if kpt_embd_coords[0].size(0) > 0:
                kpt_indices = kpt_embd_coords[i][:,2].tolist()
                if kpt_index_1 in kpt_indices and kpt_index_2 in kpt_indices:
                    sample_selection.append(i)

        # for each sample, find kpt index with maximum likelihood
        
        # best_kpts_per_sample = torch.argmax(all_kpt_attn[sample_selection], dim=1)
        # TODO only use top20 best guesses
        
        # NOTE this works because each image is flipped => distribution is fair
        # count occurences
        # kpt_max_count = torch.zeros((n_kpts,))
        # for best_kpt in best_kpts_per_sample.tolist():
        #     if best_kpt == kpt_index_1 or best_kpt == kpt_index_2:
        #         kpt_max_count[best_kpt] += 1
        # # print(kpt_max_count[kpt_index_1], kpt_max_count[kpt_index_2])
        # kpt_index_tensor = torch.argmax(kpt_max_count)

        best_sample_kpt_attn = torch.max(all_kpt_attn[sample_selection], dim=0).values
        kpt_index_tensor = torch.argmax(best_sample_kpt_attn)
    return kpt_index_tensor #, all_kpt_attn, sample_selection, best_sample_kpt_attn


def compute_max_attn(kpt_embds: torch.Tensor, test_embd: torch.Tensor):
    """ Compute maximmum attention over n samples.
    Args:
        kpt_embds: tensor of shape (n, C)
        test_embd: tensor of shape (1, C, h, w)
    """
    attn = compute_attn(kpt_embds, test_embd)
    max_attn = torch.max(attn, dim=0).values
    return max_attn


def concat_sims(sims: torch.Tensor, per_img_size = 48, rgb=False):
    n_imgs = sims.shape[0]
    out_img_size = int(np.ceil(n_imgs**0.5)) * per_img_size
    if rgb:
        out_img = np.zeros((out_img_size, out_img_size, 3), dtype=np.uint8)
    else:
        out_img = np.zeros((out_img_size, out_img_size), dtype=float)
    for i in range(out_img_size // per_img_size):
        for j in range(out_img_size // per_img_size):
            file_idx = i*(out_img_size // per_img_size)+j
            if file_idx >= n_imgs:
                break
            out_img[i*per_img_size:(i+1)*per_img_size, j*per_img_size:(j+1)*per_img_size] = F.interpolate(sims[file_idx, :, :][None, None, :, :], (per_img_size, per_img_size))[0,0].cpu().numpy()
    return out_img


def extract_local_maxima(attn: torch.Tensor, k=2, min_distance=10):
    """
    Args:
        attn: (h, w)
    Returns:
        xy_coords: attn[xy_coords[:,1], xy_coords[:,0]] = attention values of local maxima
    """
    peak_local_maxima = peak_local_max(attn.numpy(), min_distance=min_distance)
    peak_local_maxima_attn = attn[peak_local_maxima[:,0], peak_local_maxima[:,1]]
    peak_local_maxima_attn_sorted = torch.sort(peak_local_maxima_attn, descending=True)
    peak_yx_coords = peak_local_maxima[peak_local_maxima_attn_sorted.indices[:k]]
    peak_values = peak_local_maxima_attn_sorted.values[:k]
    if len(peak_local_maxima_attn_sorted.indices) == 0:
        return None, None
    if k == 1 or len(peak_local_maxima_attn_sorted.indices) == 1:
        peak_yx_coords = peak_yx_coords[None, :]
    peak_xy_coords = peak_yx_coords[:,[-1,-2]]
    return peak_xy_coords, peak_values


def extract_local_maxima_from_query(
    img_embd: torch.Tensor,
    query_embd: torch.Tensor,
    h_orig: int,
    w_orig: int,
    k=2,
    min_distance=10
):
    """
    Args:
        img_embd: (C, h, w)
        query_embd: (C,)
    """
    attn = compute_attn(query_embd[None, :], img_embd[None,:,:,:])
    attn = img_utils.inv_pad_resize_img(attn[0], h_orig, w_orig)
    local_maxima = extract_local_maxima(attn, k=k, min_distance=min_distance)
    return local_maxima, attn


def compute_kpt_attn(
    img_embd: torch.Tensor, 
    kpt_features_avg_train: torch.Tensor,
):
    kpt_attn = compute_attn(kpt_features_avg_train, img_embd[None,:,:,:])  # (n_kpts, h, w)
    kpt_attn_max = torch.max(torch.max(kpt_attn, dim=-1).values, dim=-1).values  # (n_kpts,)
    return kpt_attn, kpt_attn_max


def extract_xyz_from_attn(img_attn, img_xyz, input_img_is_padded = True, k=2):
    """
    Args:
        img_attn: (h_embd, w_embd)
        img_xyz: (h_orig, w_orig, 3)
    """
    # upscale attention to original resolution
    if input_img_is_padded:
        attn = img_utils.inv_pad_resize_img(img_attn, img_xyz.shape[0], img_xyz.shape[1])
    else:
        raise NotImplementedError
    local_maxima = extract_local_maxima(attn, k=k)
    v = img_xyz[local_maxima[0,1], local_maxima[0,0]]
    return v, local_maxima


def extract_peaks_xyz(img_attn, img_xyz_orig, max_num_peaks=5, min_distance=10, min_peak_value=0.3, debug_level=0):
    h_orig, w_orig = img_xyz_orig.shape[:2]
    attn = img_utils.inv_pad_resize_img(img_attn, h_orig, w_orig)
    peak_xy_coords, peak_values = extract_local_maxima(attn, k=max_num_peaks, min_distance=min_distance)
    if peak_xy_coords is None or peak_values is None:
        if debug_level > 0:
            print("No local maxima found")
        return torch.empty((0,3)), torch.empty((0,2)), torch.empty((0))
    peak_mask = peak_values > min_peak_value
    peak_xy_coords = peak_xy_coords[peak_mask.numpy()]
    if peak_xy_coords.shape[0] == 0:
        if debug_level > 0:
            print("No local maxima found")
        return torch.empty((0,3)), torch.empty((0,2)), torch.empty((0))
    # for depth use median in circle with radius min_distance around peak_xy_coords.
    # only consider values where attn > min_peak_value
    # TODO use function: xy_coords_to_xyz_with_correction
    peak_xyz_coords = img_xyz_orig[peak_xy_coords[:,1], peak_xy_coords[:,0]]
    for i in range(peak_xy_coords.shape[0]):
        x, y = peak_xy_coords[i]
        il,ih = max(0,y-min_distance), min(h_orig,y+min_distance)
        jl,jh = max(0,x-min_distance), min(w_orig,x+min_distance)
        candidates = img_xyz_orig[il:ih, jl:jh, :][attn[il:ih, jl:jh] > min_peak_value]
        selected_candidate_idx = torch.argmin(candidates[:,2])
        peak_xyz_coords[i, :] = candidates[selected_candidate_idx]
    return peak_xyz_coords, peak_xy_coords, peak_values[peak_mask]


def xy_coords_to_xyz_with_correction(img_attn, img_xyz_orig, query_xy_coords, thresh_rate=0.8, window_size=10):
    """
    Args:
        - img_attn: (h_embd, w_embd)
        - img_xyz_orig: (h_orig, w_orig, 3)
        - img_xy_coords: (2,)
    """
    h_orig, w_orig = img_xyz_orig.shape[:2]
    attn = img_utils.inv_pad_resize_img(img_attn, h_orig, w_orig)
    x, y = query_xy_coords
    min_attn_value = thresh_rate * attn[y,x]
    # look around (x,y) and take candidate with lowest depth
    il,ih = max(0,y-window_size), min(h_orig,y+window_size)
    jl,jh = max(0,x-window_size), min(w_orig,x+window_size)
    candidates = img_xyz_orig[il:ih, jl:jh, :][attn[il:ih, jl:jh] > min_attn_value]
    selected_candidate_idx = torch.argmin(candidates[:,2])
    query_xyz_coords = candidates[selected_candidate_idx]
    return query_xyz_coords


def kpt_img_coords_to_xyz_with_correction(
    kpt_img_coords_xy: torch.Tensor,
    kpt_labels: torch.Tensor,
    img_xyz_orig: torch.Tensor,
    img_kpt_label_likelihood: torch.Tensor,
    thresh_rate=0.8,
    window_size=10
):
    """ Transform keypoint img coords to xyz coords with *correction*
    Args:
        - kpt_img_coords: (N, 2)
        - kpt_labels: (N,)
        - img_xyz_orig: (H_orig, W_orig, 3)
        - img_kpt_label_likelihood: (K, H, W)
    """
    kpt_xyz_coords_list = []
    for i in range(kpt_img_coords_xy.shape[0]):
        kpt_xyz = xy_coords_to_xyz_with_correction(
            img_attn=img_kpt_label_likelihood[kpt_labels[i],:,:],
            img_xyz_orig=img_xyz_orig,
            query_xy_coords=kpt_img_coords_xy[i,:],
            thresh_rate=thresh_rate,
            window_size=window_size
        )
        kpt_xyz_coords_list.append(kpt_xyz)
    kpt_xyz_coords = torch.stack(kpt_xyz_coords_list, dim=0)
    return kpt_xyz_coords


def extract_peaks_xyz_for_all_kpts(kpt_attn, img_xyz_orig, max_num_peaks=5, min_distance=10, min_peak_value=0.3):
    """
    Args:
        kpt_attn: (K, H, W)
        img_xyz_orig: (h, w, 3)
    """
    k, h, w = kpt_attn.shape
    # kpt_attn = compute_attn(kpt_embds, img_embd[None,:,:,:])  # (n_kpts, h, w)

    peak_xyz_coords_all = []
    peak_xy_coords_all = []
    peak_values_all = []
    for i in range(k):
        peak_xyz_coords, peak_xy_coords, peak_values = extract_peaks_xyz(kpt_attn[i], img_xyz_orig, max_num_peaks, min_distance, min_peak_value)
        peak_xyz_coords_all.append(peak_xyz_coords)
        peak_xy_coords_all.append(peak_xy_coords)
        peak_values_all.append(peak_values)
    return peak_xyz_coords_all, peak_xy_coords_all, peak_values_all


def extract_xyz_from_attn_avg_topk(img_attn, img_xyz, input_img_is_padded = True, k=2):
    """ Take topk attention values and average their xyz values
    Args:
        img_attn: (h_embd, w_embd)
        img_xyz: (h_orig, w_orig, 3)
    """
    # upscale attention to original resolution
    if input_img_is_padded:
        attn = img_utils.inv_pad_resize_img(img_attn, img_xyz.shape[0], img_xyz.shape[1])
    else:
        raise NotImplementedError
    attn = attn.reshape(-1)
    # sort attn (descending)
    attn_sorted = torch.sort(attn, descending=True)
    # take topk
    topk_indices = attn_sorted.indices[:k]
    topk_xyz = img_xyz.reshape((-1,3))[topk_indices,:]
    attn_topk = attn_sorted.values[:k,None]
    if k > 1:
        attn_topk -= torch.min(attn_topk)
        attn_topk /= (torch.max(attn_topk) + 1e-6)
    v = torch.sum(topk_xyz * attn_topk, dim=0)
    v /= torch.norm(v)
    print(v.shape)
    return v, topk_indices


def extract_stable_keypoints(
    kpt_attn: torch.Tensor, 
    kpt_attn_max: torch.Tensor,
    img_xyz: torch.Tensor,
    stable_keypoint_indices: List[int],
    thresh_rate = 0.8,
    input_img_is_padded = True
):
    """
    Args:
        kpt_attn: (n_total_kpts, h, w)
        kpt_attn_max: (n_total_kpts,)
        img_xyz: (h, w, 3)
    """
    thresh = thresh_rate * torch.max(kpt_attn_max)
    stable_indices = np.array(stable_keypoint_indices)[kpt_attn_max[stable_keypoint_indices] >= thresh]
    stable_index_to_xyz = {}
    for kpt_index in stable_indices:
        v = extract_xyz_from_attn(kpt_attn[kpt_index], img_xyz, input_img_is_padded=input_img_is_padded)
        stable_index_to_xyz[kpt_index] = v
    return stable_indices, stable_index_to_xyz


def compute_plane_dir_from_stable_keypoints(
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
        - img_xyz: (h, w, 3)
    """
    # compute plane direction
    kpt_attn = compute_attn(kpt_features_avg_train, img_embd[None,:,:,:])  # (n_kpts, h, w)
    kpt_attn_max = torch.max(torch.max(kpt_attn, dim=-1).values, dim=-1).values  # (n_kpts,)
    
    thresh = stable_threshold_rate * torch.max(kpt_attn_max)
    plane_kpt_indices = np.array(stable_indices)[kpt_attn_max[stable_indices] >= thresh]
    if log_level > 0:
        print("plane_kpt_indices", plane_kpt_indices)
    plane_kpt_indices_set = set(plane_kpt_indices.tolist())
    # n_kpts = len(plane_kpt_indices)
    
    plane_kpt_xyz = []
    plane_kpt_idx_to_xyz = {}
    for kpt_index in plane_kpt_indices:
        # TODO adjust for higher resolution
        attn = img_utils.inv_pad_resize_img(kpt_attn[kpt_index], img_xyz.shape[0], img_xyz.shape[1])
        local_maxima = extract_local_maxima(attn, k=2)
        v = img_xyz[local_maxima[0,1], local_maxima[0,0]]
        plane_kpt_xyz.append(v)
        plane_kpt_idx_to_xyz[kpt_index] = plane_kpt_xyz[-1]

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
    plane_dir = None
    print(len(trianlge_dirs), trianlge_dirs)
    if len(trianlge_dirs) == 0:
        if log_level > 0:
            print("Warning:", "triangles have low determinant or no triangle could be constructed")
            print("plane kpt indices", plane_kpt_indices)
    else:
        plane_dir = torch.mean(torch.stack(trianlge_dirs), dim=0)
    return plane_dir

# TODO in general case we cannot use plane dir hint
# see spair_depth.ipynb for old version of construct_plane
def construct_plane_for_v1_v2(
    v1: torch.Tensor, 
    v2: torch.Tensor, 
    plane_dir_hint: torch.Tensor,
):
    plane_origin = (v1 + v2) / 2  # (3,)
    o_v1 = v1 - plane_origin
    plane_x_axis = geom_utils.construct_orthogonal_vec_torch(o_v1)
    plane_y_axis = torch.cross(o_v1, plane_x_axis)
    plane_y_axis /= torch.norm(plane_y_axis)

    plane_dir_projected = geom_utils.project_to_plane(plane_origin, plane_x_axis, plane_y_axis, plane_origin + plane_dir_hint)
    plane_normal = plane_origin + plane_dir_hint - plane_dir_projected
    plane_normal /= torch.norm(plane_normal)

    return plane_origin, plane_normal, plane_x_axis, plane_y_axis


def construct_plane_v3(
    img_xyz: torch.Tensor,
    unstable_index_pair: List[int],
    plane_dir_hint: torch.Tensor,
    kpt_attn: torch.Tensor,  # (n_kpts, h, w)
    thresh_rate = 0.8,
):
    v1, v2, plane_origin, plane_normal, plane_x_axis, plane_y_axis = None, None, None, None, None, None

    # compute plane direction
    kpt_attn_max = torch.max(torch.max(kpt_attn, dim=-1).values, dim=-1).values  # (n_kpts,)
    kpt_attn_total_max = torch.max(kpt_attn_max)
    thresh = thresh_rate * kpt_attn_total_max

    i = 0
    kpt_index_1 = unstable_index_pair[i*2]
    kpt_index_2 = unstable_index_pair[i*2 + 1]
    if kpt_attn_max[kpt_index_1] > thresh or kpt_attn_max[kpt_index_2] > thresh:
        # TODO adjust for higher depth resolution
        # => upscale attention to original resolution
        if kpt_attn_max[kpt_index_1] > kpt_attn_max[kpt_index_2]:
            attn = kpt_attn[kpt_index_1]
        else:
            attn = kpt_attn[kpt_index_2]
        attn = img_utils.inv_pad_resize_img(attn, img_xyz.shape[0], img_xyz.shape[1])
        local_maxima = extract_local_maxima(attn)
        # TODO is 0.9 a good threshold?
        if attn[local_maxima[1,1], local_maxima[1,0]] > 0.9 * attn[local_maxima[0,1], local_maxima[0,0]]:
            print("local_maxima", local_maxima)
            v1 = img_xyz[local_maxima[0,1], local_maxima[0,0]]
            v2 = img_xyz[local_maxima[1,1], local_maxima[1,0]]
            plane_origin, plane_normal, plane_x_axis, plane_y_axis = construct_plane_for_v1_v2(
                v1, v2, plane_dir_hint
            )
            
    return v1, v2, plane_origin, plane_normal, plane_x_axis, plane_y_axis


def determine_right_left(
    v1, v2, plane_origin, plane_normal
):
    # determine right left
    v1_sign = torch.dot(plane_normal, (v1 - plane_origin))
    v2_sign = torch.dot(plane_normal, (v2 - plane_origin))
    return v1_sign > v2_sign


# =======================================
# orientation with depth estimation
# =======================================


# def compute_orientation_with_normal_info(v1_xyz, v2_xyz, stable_points_xyz, stable_point_normals):
#     n = stable_points_xyz.shape[0]
#     stable_point_norms = torch.norm(stable_points_xyz, dim=1, keepdim=False)
#     v1v2 = v2_xyz - v1_xyz

#     # TODO this is inefficient => better compute mask first for points that are valid
#     triangle_normals = []
#     for i in range(n):
#         vi = stable_points_xyz[i]
#         triangle_normal = torch.zeros(3)
#         if stable_point_norms[i] > 1e-9:
#             v1vi = vi - v1_xyz
#             triangle_normal = torch.cross(v1v2, v1vi)
#         triangle_normals.append(triangle_normal)
#     triangle_normals = torch.stack(triangle_normals, dim=0)
#     triangle_normal_norms = torch.norm(triangle_normals, dim=1, keepdim=True)
#     triangle_normals = triangle_normals / (triangle_normal_norms + 1e-6)

#     # project edges on triangle normals
#     projections = triangle_normals.float() @ stable_point_normals.float().T
#     projections[:, stable_point_norms < 1e-6] = 0
#     projections[triangle_normal_norms[:,0] < 1e-6, :] = 0

#     return projections

# def compute_orientation_with_normal_info_v2(v1_xyz, v2_xyz, stable_points_xyz_1, stable_point_normals_1, stable_points_xyz_2, stable_point_normals_2):
#     n = stable_point_normals_1.shape[0]
#     stable_point_norms_1 = torch.norm(stable_point_normals_1, dim=1, keepdim=False)
#     stable_point_norms_2 = torch.norm(stable_point_normals_2, dim=1, keepdim=False)

#     v1v2 = (v2_xyz - v1_xyz)[None,:]
#     # v1s2 = stable_points_xyz_2 - v1_xyz[None,:]

#     v1s1 = stable_points_xyz_1 - v1_xyz
#     # v1s2 = stable_points_xyz_2 - v1_xyz
#     mask = (stable_point_norms_1 > 1e-9) & (stable_point_norms_2 > 1e-9)
#     triangle_normals = torch.zeros((n, 3), dtype=torch.double)
#     if torch.sum(mask) > 0:
#         # triangle_normals[mask] = torch.cross(v1s1[mask], v1s2[mask], dim=-1)
#         triangle_normals[mask] = torch.cross(v1s1[mask], v1v2, dim=-1)
#     triangle_normal_norms = torch.norm(triangle_normals, dim=1, keepdim=True)
#     triangle_normals[mask] /= (triangle_normal_norms[mask] + 1e-6)
#     stable_point_normals_avg = (stable_point_normals_1 + stable_point_normals_2) / 2
#     stable_point_normals_avg[~mask] = 0
#     stable_point_normals_avg[mask] /= torch.norm(stable_point_normals_avg[mask], dim=1, keepdim=True)
#     # project edges on triangle normals
#     projections = triangle_normals.float() @ stable_point_normals_avg.float().T

#     return projections


def compute_orientation_with_normal_info_v2(
    v1_xyz: torch.Tensor,
    stable_points_xyz_1: torch.Tensor,
    stable_point_normals_1: torch.Tensor,
    stable_points_xyz_2: torch.Tensor,
    stable_point_normals_2: torch.Tensor
):
    """ Compute orientation based 
    Args:
        v1_xyz: (3,)
        stable_points_xyz_1: (n, 3)
        stable_point_normals_1: (n, 3)
        stable_points_xyz_2: (n, 3)
        stable_point_normals_2: (n, 3)
    Returns:
        angle_triangle_normal_to_stable_point_cross: (n,)
    """
    n = stable_point_normals_1.shape[0]
    stable_point_norms_1 = torch.norm(stable_point_normals_1, dim=1, keepdim=False)
    stable_point_norms_2 = torch.norm(stable_point_normals_2, dim=1, keepdim=False)

    v1s1 = stable_points_xyz_1 - v1_xyz
    v1s2 = stable_points_xyz_2 - v1_xyz
    mask = (stable_point_norms_1 > 1e-9) & (stable_point_norms_2 > 1e-9)
    triangle_normals = torch.zeros((n, 3), dtype=torch.double)
    if torch.sum(mask) > 0:
        triangle_normals[mask] = torch.cross(v1s1[mask], v1s2[mask], dim=-1)
    triangle_normal_norms = torch.norm(triangle_normals, dim=1, keepdim=True)
    triangle_normals[mask] /= (triangle_normal_norms[mask] + 1e-6)
    stable_point_normals_avg = (stable_point_normals_1 + stable_point_normals_2) / 2
    stable_point_normals_avg[~mask] = 0
    stable_point_normals_avg[mask] /= torch.norm(stable_point_normals_avg[mask], dim=1, keepdim=True)
    
    # s1s2 will be our plane normal
    s1s2 = stable_points_xyz_1 - stable_points_xyz_2
    s1s2[mask] /= torch.norm(s1s2[mask], dim=1, keepdim=True)
    stable_point_cross = torch.zeros_like(stable_point_normals_avg)
    stable_point_cross[mask] = torch.cross(stable_point_normals_avg[mask], s1s2[mask], dim=-1)

    # angle with from triangle normal to stable point cross rotated along s2s1
    angle_triangle_normal_to_stable_point_cross = torch.zeros((n), dtype=torch.double)
    angle_triangle_normal_to_stable_point_cross[mask] = torch.atan2(
        torch.sum(torch.cross(stable_point_cross[mask], triangle_normals[mask], dim=1) * s1s2[mask], dim=1),
        torch.sum(triangle_normals[mask] * stable_point_cross[mask], dim=1)
    )

    return angle_triangle_normal_to_stable_point_cross





def compute_projection_for_img_with_normal_info(
    kpt_index_1,
    kpt_index_2,
    img_embd,
    img_xyz,
    img_normals,
    kpt_img_coords_gt: torch.Tensor,
    kpt_features_avg_train: torch.Tensor,
    # stable_indices: List[int],

    # TODO extend to multiple stable index pairs
    # stable_index_pair: List[int],
    stable_indices_1: List[int],
    stable_indices_2: List[int],
    thresh_rate=0.8,
    input_img_is_padded=True,
):
    img_kpt_indices = kpt_img_coords_gt[:,2]
    kpt_index_1_mask = img_kpt_indices == kpt_index_1
    kpt_index_2_mask = img_kpt_indices == kpt_index_2

    # if both keypoints are not present in the image, return None
    kpt_1_in_img = torch.sum(kpt_index_1_mask) > 0
    kpt_2_in_img = torch.sum(kpt_index_2_mask) > 0
    if not kpt_1_in_img and not kpt_2_in_img:
        return None, None, None, None, None, None, None, None
    
    # TODO should we gt data?
    kpt_attn, kpt_attn_max = compute_kpt_attn(
        img_embd,
        kpt_features_avg_train=kpt_features_avg_train
    )
    thresh = thresh_rate * torch.max(kpt_attn_max)

    # we need at least one pair of stable points
    m1 = kpt_attn_max[stable_indices_1] >= thresh
    m2 = kpt_attn_max[stable_indices_1] >= thresh
    m12 = m1 & m2
    if torch.sum(m12) == 0:
        return None, None, None, None, None, None, None, None

    # extract xyz and normal for keypoint indices that are above the threshold
    # TODO implement function: ft_align_utils.extract_stable_keypoints
    stable_xyz = torch.zeros((30, 3), dtype=torch.double)
    stable_normals = torch.zeros((30, 3), dtype=torch.double)
    stable_indices = list(set(stable_indices_1).union(set(stable_indices_2)))
    stable_indices_above_thresh = torch.tensor(stable_indices)[kpt_attn_max[stable_indices] >= thresh]
    for kpt_index in stable_indices_above_thresh:
        v, local_maxima = extract_xyz_from_attn(kpt_attn[kpt_index], img_xyz, input_img_is_padded=input_img_is_padded)
        stable_xyz[kpt_index] = v
        # stable_xyz[kpt_index_to_stable_index[kpt_index]], _ = ft_align_utils.extract_xyz_from_attn_avg_topk(
        #     kpt_attn[kpt_index], img_xyz, k=5, input_img_is_padded=input_img_is_padded)
        # stable_normals[kpt_index_to_stable_index[kpt_index]] = img_normals[local_maxima[0,1], local_maxima[0,0]]
        stable_normals[kpt_index], _ = extract_xyz_from_attn_avg_topk(
            kpt_attn[kpt_index], img_normals, k=5, input_img_is_padded=input_img_is_padded)
    stable_points_xyz_1 = stable_xyz[stable_indices_1,:]
    stable_point_normals_1 = stable_normals[stable_indices_1,:]
    stable_points_xyz_2 = stable_xyz[stable_indices_2,:]
    stable_point_normals_2 = stable_normals[stable_indices_2,:]

    # compute orientation (angle) for kpt_index_1 and kpt_index_2
    v1_img_coords = kpt_img_coords_gt[kpt_index_1_mask][0,:2]
    v2_img_coords = kpt_img_coords_gt[kpt_index_2_mask][0,:2]
    v1_xyz = img_xyz[v1_img_coords[1], v1_img_coords[0]]
    v2_xyz = img_xyz[v2_img_coords[1], v2_img_coords[0]]
    projections_1, projections_2 = None, None
    if kpt_1_in_img:
        projections_1 = compute_orientation_with_normal_info_v2(
            v1_xyz, stable_points_xyz_1, stable_point_normals_1, stable_points_xyz_2, stable_point_normals_2
        )
    if kpt_2_in_img:
        projections_2 = compute_orientation_with_normal_info_v2(
            v2_xyz, stable_points_xyz_1, stable_point_normals_1, stable_points_xyz_2, stable_point_normals_2
        )
    return projections_1, projections_2, v1_xyz, v2_xyz, stable_points_xyz_1, stable_point_normals_1, stable_points_xyz_2, stable_point_normals_2


def compute_orientation_with_normal_info_v3(
    xyz_candidates: torch.Tensor,
    stable_points_xyz_1: torch.Tensor,
    stable_point_normals_1: torch.Tensor,
    stable_points_xyz_2: torch.Tensor,
    stable_point_normals_2: torch.Tensor
):
    """ Compute orientation based 
    Args:
        xyz_candidates: (n_candidates, 3)
        stable_points_xyz_1: (n_stable, 3)
        stable_point_normals_1: (n_stable, 3)
        stable_points_xyz_2: (n_stable, 3)
        stable_point_normals_2: (n_stable, 3)
    Returns:
        angle_triangle_normal_to_stable_point_cross: (n,)
    """
    n_candidates = xyz_candidates.shape[0]
    n_stable = stable_point_normals_1.shape[0]
    stable_point_norms_1 = torch.norm(stable_point_normals_1, dim=-1, keepdim=False)
    stable_point_norms_2 = torch.norm(stable_point_normals_2, dim=-1, keepdim=False)

    cs1 = stable_points_xyz_1[:,None,:] - xyz_candidates[None,:,:]  # (n_stable, n_candidates, 3)
    cs2 = stable_points_xyz_2[:,None,:] - xyz_candidates[None,:,:]  # (n_stable, n_candidates, 3)
    mask = (stable_point_norms_1 > 1e-9) & (stable_point_norms_2 > 1e-9)
    triangle_normals = torch.zeros((n_stable, n_candidates, 3), dtype=torch.double)
    if torch.sum(mask) > 0:
        triangle_normals[mask] = torch.cross(cs1[mask], cs2[mask], dim=-1)
    triangle_normal_norms = torch.norm(triangle_normals, dim=-1, keepdim=True)
    triangle_normals[mask] /= (triangle_normal_norms[mask] + 1e-6)
    stable_point_normals_avg = (stable_point_normals_1 + stable_point_normals_2) / 2
    stable_point_normals_avg[~mask] = 0
    stable_point_normals_avg[mask] /= torch.norm(stable_point_normals_avg[mask], dim=-1, keepdim=True)
    
    # s2s1 will be our plane normal
    s2s1 = stable_points_xyz_1 - stable_points_xyz_2
    s2s1[mask] /= torch.norm(s2s1[mask], dim=-1, keepdim=True)
    stable_point_cross = torch.zeros_like(stable_point_normals_avg)
    stable_point_cross[mask] = torch.cross(stable_point_normals_avg[mask], s2s1[mask], dim=-1)

    # angle with from triangle normal to stable point cross rotated along s2s1
    angle_triangle_normal_to_stable_point_cross = torch.zeros((n_stable, n_candidates), dtype=torch.double)
    angle_triangle_normal_to_stable_point_cross[mask] = torch.atan2(
        torch.sum(torch.cross(stable_point_cross[mask][:,None,:], triangle_normals[mask], dim=-1) * s2s1[mask][:,None,:], dim=-1),
        torch.sum(triangle_normals[mask] * stable_point_cross[mask][:,None,:], dim=-1)
    )

    return angle_triangle_normal_to_stable_point_cross


def compute_similarity_with_normal_info(
    q_embd: torch.Tensor,
    test_embd: torch.Tensor,
):
    q_attn = compute_attn(q_embd, test_embd)

    # extract blobs from q_attn




    pass
