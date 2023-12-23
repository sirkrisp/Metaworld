import numpy as np

import keyframes.policies.agent as agent
from metaworld.policies.policy import Policy, assert_fully_parsed, move
import numpy as np


"""
This class implements a push policy. NOTE we never push up, only down.

"""


class PushPolicy(agent.Agent):
    def __init__(self, push_start_pos: np.ndarray, target_pos: np.ndarray, log=False, log_level=0):
        super().__init__()

        self.push_start_pos = push_start_pos
        self.target_pos = target_pos
        self.push_dir = self.target_pos - self.push_start_pos
        self.push_start_offset = -0.01 * self.push_dir / np.linalg.norm(self.push_dir)

        # Steps:
        # -2: Wait
        # -1: open/close gripper based on push direction and move to push start pos z
        # 0: move on top of next to push start pos (to avoid collision with object)
        # 1: move next to push start pos
        # 2: move in push direction
        self.step = -2
        # TODO path points should be one of MoveTo, Gripper, and Wait
        self.path_points = [
            self.push_start_pos + self.push_start_offset + np.array([0.0, 0.0, 0.2]),
            self.push_start_pos + self.push_start_offset,
            self.target_pos,
        ]
        for i in range(len(self.path_points)):
            self.path_points[i][2] = np.min([self.path_points[i][2], 0.4])
        self.log = log
        self.log_level = log_level
        
        self.push_index = np.argmax(np.abs(self.push_dir))
        self.open_gripper = self.push_index == 1 or self.push_index == 0  # if we push in y direction, we need to open gripper

        self.num_gripper_steps = 10
        self.curr_gripper_step = 0
        self.num_wait_steps = 40
        self.curr_wait_step = 0
        
    def _log(self, *msgs, log_level=0):
        if self.log and log_level <= self.log_level:
            print(msgs)

    def get_action(self, obs):
        
        curr_hand_pos = obs[:3]
        action = np.zeros(4)
        err_threshold = 0.01
        to_xyz = curr_hand_pos.copy()


        if self.step == -2:
            # wait
            self.curr_wait_step += 1
            if self.curr_wait_step == self.num_wait_steps:
                self.step = 0
                self._log("proceed with step 0")
        if self.step == -1:
            # open/close gripper and ascend to push start pos z
            action[3] = -1.0 if self.open_gripper else 1.0
            self.curr_gripper_step += 1
            to_xyz[2] = self.path_points[0][2]
            if (self.curr_gripper_step >= self.num_gripper_steps) and (np.linalg.norm(to_xyz - curr_hand_pos) < err_threshold):
                self.step = 1
                self._log("proceed with step 1")
        if self.step >=0 and self.step < len(self.path_points):
            action[3] = -1.0 if self.open_gripper else 1.0
            # move to next path point
            to_xyz = self.path_points[self.step]
            if np.linalg.norm(to_xyz - curr_hand_pos) < err_threshold:
                self.step += 1
                self._log("proceed with step {}".format(self.step))

        action[:3] = move(curr_hand_pos, to_xyz=to_xyz, p=10.0)

        self._log("action", action, log_level=1)

        return action
