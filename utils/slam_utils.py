from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd, numpy_image_to_torch
import numpy as np


def get_super_glue_extractor(device: str = "cuda"):
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    return extractor


def compute_features(extractor: SuperPoint, image_np: np.ndarray, device: str = "cuda"):
    image_torch = numpy_image_to_torch(image_np).to(device)
    return rbd(extractor.extract(image_torch))