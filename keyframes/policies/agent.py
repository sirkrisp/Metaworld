import numpy as np
import keyframes.policies.policy_path_points as policy_path_points


class Agent:

    def __init__(self, path_points: list[policy_path_points.PathPoint], log=False, log_level=0):
        self.path_points = path_points
        self.curr_path_point = 0
        self.curr_path_point_done = False
        self.log = log
        self.log_level = log_level

    def get_action(self, obs):
        if self.curr_path_point_done:
            self.curr_path_point += 1
            self.curr_path_point_done = False
        if self.curr_path_point >= len(self.path_points):
            return np.zeros(4)
        action = self.path_points[self.curr_path_point].get_action(obs)
        self.curr_path_point_done = self.path_points[self.curr_path_point].is_done(obs)
        self._log(f"curr_path_point: {self.curr_path_point}, curr_path_point_done: {self.curr_path_point_done}, action: {action}")
        return action

    def reset(self):
        self.curr_path_point = 0
        self.curr_path_point_done = False
        for path_point in self.path_points:
            path_point.reset()

    def _log(self, *msgs, log_level=0):
        if self.log and log_level <= self.log_level:
            print(msgs)
    