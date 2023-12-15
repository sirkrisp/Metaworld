import numpy as np


class PickPlacePolicy:

    def __init__(self, grasp_point: np.ndarray, target_point: np.ndarray) -> None:
        self.grasp_point = grasp_point
        self.target_point = target_point

    def get_action(self):
        pass