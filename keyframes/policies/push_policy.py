""" This module implements a push policy. """
import numpy as np
import keyframes.policies.agent as agent
import keyframes.policies.policy_path_points as policy_path_points


class PushPolicy(agent.Agent):
    """
    This class implements a push policy. NOTE we never push up, only down.
    """

    def __init__(
        self,
        push_start_pos: np.ndarray,
        push_goal_pos: np.ndarray,
        log=False,
        log_level=0,
    ):
        push_dir = push_goal_pos - push_start_pos

        # if we push in y direction, we need to open gripper
        push_index = np.argmax(np.abs(push_dir))
        open_gripper = push_index == 1 or push_index == 0
        gripper_action = -1.0 if open_gripper else 1.0

        push_start_offset = (-0.09 if open_gripper else -0.01) * push_dir / np.linalg.norm(push_dir)

        path_pts = [
            push_start_pos + push_start_offset + np.array([0.0, 0.0, 0.2]),
            push_start_pos + push_start_offset,
            push_goal_pos,
        ]
        for i, pt in enumerate(path_pts):
            path_pts[i][2] = np.min([pt[2], 0.4])
        move_up_z = path_pts[0][2]     

        threshold = 0.01
        p = 10.0

        path_points = [
            # Wait
            policy_path_points.Wait(num_wait_steps=40),
            # Move up (or down depending on current ee position)
            policy_path_points.MoveAxis(
                axis=2,
                value=move_up_z,
                gripper_action=gripper_action,
                p=p,
                threshold=threshold,
                max_steps=10,
            ),
            *[
                policy_path_points.MoveTo(
                    pos=pt,
                    gripper_action=gripper_action,
                    p=p,
                    threshold=threshold,
                    max_steps=40,
                )
                for pt in path_pts
            ],
        ]

        super().__init__(path_points=path_points, log=log, log_level=log_level)

    # def _log(self, *msgs, log_level=0):
    #     if self.log and log_level <= self.log_level:
    #         print(msgs)

    # def get_action(self, obs):
    #     curr_hand_pos = obs[:3]
    #     action = np.zeros(4)
    #     err_threshold = 0.01
    #     to_xyz = curr_hand_pos.copy()

    #     if self.step == -2:
    #         # wait
    #         self.curr_wait_step += 1
    #         if self.curr_wait_step == self.num_wait_steps:
    #             self.step = 0
    #             self._log("proceed with step 0")
    #     if self.step == -1:
    #         # open/close gripper and ascend to push start pos z
    #         action[3] = -1.0 if self.open_gripper else 1.0
    #         self.curr_gripper_step += 1
    #         to_xyz[2] = self.path_points[0][2]
    #         if (self.curr_gripper_step >= self.num_gripper_steps) and (
    #             np.linalg.norm(to_xyz - curr_hand_pos) < err_threshold
    #         ):
    #             self.step = 1
    #             self._log("proceed with step 1")
    #     if self.step >= 0 and self.step < len(self.path_points):
    #         action[3] = -1.0 if self.open_gripper else 1.0
    #         # move to next path point
    #         to_xyz = self.path_points[self.step]
    #         if np.linalg.norm(to_xyz - curr_hand_pos) < err_threshold:
    #             self.step += 1
    #             self._log("proceed with step {}".format(self.step))

    #     action[:3] = move(curr_hand_pos, to_xyz=to_xyz, p=10.0)

    #     self._log("action", action, log_level=1)

    #     return action
