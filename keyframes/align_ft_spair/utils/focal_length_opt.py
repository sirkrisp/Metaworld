""" Optimise focal length for each image by minimizing variance of keypoint ratios/angles in dataset
"""

from typing import Optional, List
import torch
import numpy as np
from keyframes.align_ft_spair.utils import ft_align_utils, spair_utils, geom_utils, kpt_likelihood_utils
from utils import torch_utils
from tqdm import tqdm
from sklearn.cluster import KMeans



