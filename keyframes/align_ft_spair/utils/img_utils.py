import torch
import torch.nn as nn
import torch.nn.functional as F


# 1.2) pad and resize depth
def get_pad_sizes(x):
    max_size = max(x.shape[0], x.shape[1])
    return max_size - x.shape[0], max_size - x.shape[1]

def get_pad_sizes_from_img_shape(h, w):
    max_size = max(h, w)
    return max_size - h, max_size - w

def to_torch(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)

def zero_pad_square(x):
    pad_widths = get_pad_sizes(x)
    return F.pad(to_torch(x)[None,None,:,:], [0,0,pad_widths[0]//2,pad_widths[0]//2, pad_widths[1]//2, pad_widths[1]//2], value=0)[0,0]

# TODO implement forward function as well
def inv_pad_resize_img(img_resized_padded: torch.Tensor, orig_img_h, orig_img_w):
    """
    Args:
        img_resized_padded: (H, W) tensor where H == W and has been padded to be square
    """
    pad_h, pad_w = get_pad_sizes_from_img_shape(orig_img_h, orig_img_w)
    max_size = max(orig_img_h, orig_img_w)
    # upsample img
    img_max = nn.Upsample(size=(max_size, max_size), mode='bilinear')(img_resized_padded[None, None, :, :])[0,0]  # shape (img_size, img_size)
    # crop img
    img_cropped = img_max[pad_h//2:pad_h//2+orig_img_h, pad_w//2:pad_w//2+orig_img_w]
    return img_cropped