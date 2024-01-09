from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


class SegAny:

    def __init__(self, ckpt_path, img_shape) -> None:
        sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
        self.predictor = SamPredictor(sam)
        self.img_shape = img_shape
        self.x = np.linspace(0, img_shape[0] - 1, img_shape[0])
        self.y = np.linspace(0, img_shape[1] - 1, img_shape[1])

    def _points_in_mask(self, pixel_coords_xy: np.ndarray, mask: np.ndarray):
        """
        Args:
            pixel_coords_xy: (N, 2)
            mask: (H, W)
        """
        interp_mask = RegularGridInterpolator((self.x, self.y), mask.astype(float), bounds_error=False, fill_value=0)

        pixel_coords_xy_center = np.floor(pixel_coords_xy)
        offsets = np.array([[0,0], [0,1], [1,0], [1,1], [-1,0], [0,-1], [-1,-1], [1,-1], [-1,1]])
        pixel_coords_xy_region = pixel_coords_xy_center[:, np.newaxis, :] + offsets[np.newaxis, :, :] # shape (n, 9, 2)

        pixel_mask_value = np.zeros(pixel_coords_xy.shape[0], dtype=np.float32)
        for i in range(9):
            pixel_mask_value_i = interp_mask(pixel_coords_xy_region[:, i, :] @ np.array([[0, 1], [1, 0]]))  # shape (n,)
            # take min depth across all 9 neighbours
            pixel_mask_value = np.maximum(pixel_mask_value, pixel_mask_value_i)  # shape (n,)

        return pixel_mask_value > 0

    # TODO mask_is_valid makes sure that seg any does not output a mask that covers the entire image
    # However this check is very hacky and should be replaced with something more robust
    def mask_is_valid(self, mask: np.ndarray):
        return mask.sum() < 0.3 * mask.shape[0] * mask.shape[1]
        
    def compute_mask(self, img: np.ndarray, points: np.ndarray, subset_fraction=2, n_iters=10):
        """ Compute object mask from image and points.
        Args:
            img: (H, W, 3)
            points: (N, 2)
        Returns:
            mask: (H, W) np.ndarray
        """
        self.predictor.set_image(img)

        n = points.shape[0]
        # NOTE we set n_sample_points to 1 to make sure we only select one object
        # (points could belong to different objects, we want to find the object that covers most points)
        n_sample_points = 1 # int(np.ceil(n / subset_fraction))
        input_labels = np.ones(n_sample_points)

        best_mask = None
        n_points_in_best_mask = 0
        for _ in range(n_iters):
            subset = np.random.choice(n, n_sample_points, replace=False)
            input_points = points[subset]
            masks, mask_scores, logits = self.predictor.predict(
                point_coords=input_points, 
                point_labels=input_labels
            )
            for mask in masks:
                if not self.mask_is_valid(mask):
                    continue
                n_points_in_mask = self._points_in_mask(points, mask).sum()
                if n_points_in_mask > n_points_in_best_mask:
                    best_mask = mask
                    n_points_in_best_mask = self._points_in_mask(points, mask).sum()

        return best_mask


# ====================
# Utils
# ====================

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  