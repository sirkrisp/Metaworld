from re import M
import numpy as np
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class PathPoint:
    def __init__(self, type: str, max_steps: int = 100):
        self.type = type
        self.curr_step = 0
        self.max_steps = max_steps

    def _reached_max_steps(self):
        return self.curr_step >= self.max_steps
    
    def _increment_step(self):
        self.curr_step += 1

    def is_done(self, obs: np.ndarray):
        raise NotImplementedError
        return self.curr_step >= self.max_steps

    def get_action(self, obs: np.ndarray):
        raise NotImplementedError

    def reset(self):
        self.curr_step = 0

    def __str__(self):
        return self.type


class MoveTo(PathPoint):
    def __init__(self, pos: np.ndarray, gripper_action: float = 0.0, p=10.0, threshold=0.01, max_steps=100):
        """
        Args:
            pos: target position
            gripper_action: 1 to close and -1 to open
            p: proportional gain
            threshold: distance threshold to target position
        """
        super().__init__("MoveTo", max_steps=max_steps)
        self.pos = pos
        self.gripper_action = gripper_action
        self.p = p
        self.threshold = threshold

    def is_done(self, obs: np.ndarray):
        curr_pos = obs[:3]
        return (np.linalg.norm(self.pos - curr_pos) < self.threshold) or self._reached_max_steps()

    def get_action(self, obs: np.ndarray):
        self._increment_step()
        curr_pos = obs[:3]
        action = np.zeros(4)
        action[:3] = move(curr_pos, self.pos, p=self.p)
        action[3] = self.gripper_action
        return action


class MoveAxis(PathPoint):
    def __init__(
        self, axis: int, value: float, gripper_action: float = 0.0, p=10.0, threshold=0.01, max_steps=100
    ):
        """
        Args:
            axis: 0, 1, or 2 for x, y, or z axis
            value: target position on axis
            gripper_action: 1 to close and -1 to open
            p: proportional gain
            threshold: distance threshold to target position
        """
        super().__init__("MoveUp", max_steps=max_steps)
        self.axis = axis
        self.value = value
        self.gripper_action = gripper_action
        self.p = p
        self.threshold = threshold

    def is_done(self, obs: np.ndarray):
        curr_pos = obs[:3]
        return (np.abs(self.value - curr_pos[self.axis]) < self.threshold) or self._reached_max_steps()

    def get_action(self, obs: np.ndarray):
        self._increment_step()
        curr_pos = obs[:3]
        to_xyz = curr_pos.copy()
        to_xyz[self.axis] = self.value
        action = np.zeros(4)
        action[:3] = move(curr_pos, to_xyz, p=self.p)
        action[3] = self.gripper_action
        return action


class Gripper(PathPoint):
    def __init__(self, gripper_action: float, num_gripper_steps=10, max_steps=100):
        """
        Args:
            gripper_action: 1 to close and -1 to open
            num_gripper_steps: number of steps to close/open gripper
        """
        super().__init__("Gripper", max_steps=max_steps)
        self.gripper_action = gripper_action
        self.num_gripper_steps = num_gripper_steps
        self.curr_gripper_step = 0

    def is_done(self, obs: np.ndarray):
        return (self.curr_gripper_step == self.num_gripper_steps) or self._reached_max_steps()

    def get_action(self, obs: np.ndarray):
        self._increment_step()
        action = np.zeros(4)
        action[3] = self.gripper_action
        self.curr_gripper_step += 1
        return action

    def reset(self):
        super().reset()
        self.curr_gripper_step = 0


class Wait(PathPoint):
    def __init__(self, num_wait_steps=40, max_steps=100):
        """
        Args:
            num_wait_steps: number of steps to wait
        """
        super().__init__("Wait", max_steps=max_steps)
        self.num_wait_steps = num_wait_steps
        self.curr_wait_step = 0

    def is_done(self, obs: np.ndarray):
        return (self.curr_wait_step == self.num_wait_steps) or self._reached_max_steps()

    def get_action(self, obs: np.ndarray):
        self._increment_step()
        self.curr_wait_step += 1
        return np.zeros(4)

    def reset(self):
        super().reset()
        self.curr_wait_step = 0
