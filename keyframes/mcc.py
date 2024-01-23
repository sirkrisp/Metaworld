from dataclasses import dataclass
from typing import Any
import models.extern.mcc_model as mcc_model
import utils.mcc_misc_utils as mcc_misc_utils
import utils.mcc_data_utils as mcc_data_utils
from skimage import measure
import torch
import numpy as np



class ReconstructMCC:

    def __init__(
            self, 
            mcc_ckpt_path: str,
            device="cuda", 
    ) -> None:
        self.device = device
        self.mcc_model = mcc_misc_utils.load_model_from_ckpt(ckpt_path=mcc_ckpt_path, device=device)
        self.model_input = None
        self.masked_mean = None
        self.masked_max_dist = None

    def set_model_input(self, img, seg, xyz, grid_granularity=0.025):
        """
        Args:
            img: (3, H, W) np.ndarray
            seg: (H, W) np.ndarray
            xyz: (H*W, 3) np.ndarray
        """
        self.masked_mean, self.masked_max_dist = mcc_data_utils.compute_masked_mean_and_max_dist(img, xyz, seg)
        self.model_input = mcc_data_utils.get_model_input(
            img, 
            xyz.reshape(img.shape), 
            seg, 
            grid_size=2, # constant because xyz will be normalized to -1 to 1
            grid_granularity=grid_granularity, 
            device=self.device
        )
        for k, v in self.model_input.items():
            self.model_input[k] = v.to(self.device)

    def query_model(self, normalized_query_points):
        """
        Args:
            normalized_query_points: (N, 3) torch.Tensor
        Returns:
            pred_occupy: (N) torch.Tensor
            pred_colors: (N, 3) torch.Tensor
        """
        assert self.model_input is not None, "Please set input data first."
        is_seen = torch.zeros((normalized_query_points.shape[0]), device=self.device).bool()
        unseen_rgb = torch.zeros_like(normalized_query_points.float(), device=self.device)
        model_input = {
            "seen_xyz": self.model_input["seen_xyz"],
            "seen_rgb": self.model_input["seen_rgb"],
            "unseen_xyz": normalized_query_points[None, ...].float(),  # query points
            "unseen_rgb": unseen_rgb[None, ...],  # only used for loss => not needed for inference
            "is_seen": is_seen[None, ...],  # only used for loss => not needed for inference
            "is_valid": self.model_input["is_valid"],  # only used for loss => not needed for inference
        }
        for k, v in model_input.items():
            model_input[k] = v.to(self.device)
        # TODO pred_occupy has batch dim but pred_colors does not. => fix this
        pred_occupy, pred_colors = mcc_misc_utils.predict_mcc(self.mcc_model, model_input)
        return pred_occupy[0], pred_colors
    
    def compute_occupy_threshold(self):
        assert self.model_input is not None, "Please set input data first."
        query_points = self.model_input["seen_xyz"][self.model_input["is_valid"]]
        print(query_points.shape)
        pred_occupy, pred_colors = self.query_model(query_points)
        pred_occupy = pred_occupy.cpu().numpy().reshape(-1)
        # TODO this is not so robust
        return np.min(pred_occupy)
        # return 0.95 * (np.median(pred_occupy) - np.std(pred_occupy))

    def reconstruct(self):
        """
        Returns:
            pred_occupy: ((2/grid_granularity)**3) torch.Tensor
            pred_colors: ((2/grid_granularity)**3, 3) torch.Tensor
            occupy_threshold: float
            verts: (n_verts, 3) np.ndarray
            faces: (n_faces, 3) np.ndarray
            vert_normals: (n_verts, 3) np.ndarray
            values: (N, 3) np.ndarray
        """
        assert self.model_input is not None, "Please set input data first."
        # TODO pred_occupy has batch dim but pred_colors does not. => fix this
        pred_occupy, pred_colors = mcc_misc_utils.predict_mcc(self.mcc_model, self.model_input)
        pred_occupy = pred_occupy[0]
        # get occupy threshold from querying seen points
        occupy_threshold = self.compute_occupy_threshold()
        n_cube = int(np.round(pred_occupy.shape[0] ** (1/3)))
        verts, faces, vert_normals, values = measure.marching_cubes(-pred_occupy.cpu().numpy().reshape(n_cube,n_cube,n_cube), level=-occupy_threshold)
        verts = verts - n_cube/2
        verts *= 2 / n_cube  # grid_size is 2
        return pred_occupy, pred_colors, occupy_threshold, verts, faces, vert_normals, values
    
    def normalize(self, xyz):
        assert self.masked_mean is not None, "Please set input data first."
        return (xyz - self.masked_mean) / self.masked_max_dist

    def denormalize(self, xyz):
        assert self.masked_mean is not None, "Please set input data first."
        return xyz * self.masked_max_dist + self.masked_mean

