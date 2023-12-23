import numpy as np

import keyframes.policies.agent as agent
from metaworld.policies.policy import Policy, assert_fully_parsed, move
import numpy as np
import keyframes.policies.policy_path_points as policy_path_points


"""
This class implements a pick and place policy.
It was copied from osil/inference/agent_v4_grasp
"""


class PickPlacePolicy(agent.Agent):
    def __init__(
        self, grasp_pos: np.ndarray, target_pos: np.ndarray, log=False, log_level=0
    ):
        correction = np.array([0.0, 0.0, -0.01])
        top_pos_offset = np.array([0.0, 0.0, 0.2])
        grasp_pos = grasp_pos + correction
        target_pos = target_pos + correction

        move_up_z = 0.1
        max_ee_pos = np.array([1, 1, 0.4])

        threshold = 0.01
        p = 10.0

        path_points = [
            # Wait
            policy_path_points.Wait(num_wait_steps=40),
            # Move up (or down depending on current ee position)
            policy_path_points.MoveAxis(
                axis=2, value=move_up_z, gripper_action=0.0, p=p, threshold=threshold, max_steps=10
            ),
            # Move above grasp position
            policy_path_points.MoveTo(
                pos=np.minimum(max_ee_pos, grasp_pos + top_pos_offset),
                gripper_action=0.0,
                p=p,
                threshold=threshold,
            ),
            # Move to grasp position
            policy_path_points.MoveTo(
                pos=grasp_pos, gripper_action=0.0, p=p, threshold=threshold
            ),
            # Grasp
            policy_path_points.Gripper(gripper_action=1.0, num_gripper_steps=10),
            # Move up (or down depending on current ee position)
            policy_path_points.MoveAxis(
                axis=2, value=move_up_z, gripper_action=1.0, p=p, threshold=threshold, max_steps=10
            ),
            # policy_path_points.MoveTo(
            #     pos=np.minimum(max_ee_pos, grasp_pos + top_pos_offset),
            #     gripper_action=1.0,
            #     p=p,
            #     threshold=threshold,
            # ),
            # Move above grasp position
            policy_path_points.MoveTo(
                pos=np.minimum(max_ee_pos, target_pos + top_pos_offset), 
                gripper_action=1.0, 
                p=p, 
                threshold=threshold
            ),
            # Move above goal position
            policy_path_points.MoveTo(
                pos=target_pos, gripper_action=1.0, p=p, threshold=threshold
            ),
        ]

        super().__init__(path_points=path_points, log=log, log_level=log_level)

         # Steps:
        # -1: Wait
        # 0: move above grasp position
        # 1: move to grasp position
        # 2: grasp
        # 3: move above grasp position
        # 4: move above goal position
        # 5: move to goal position
        # NOTE sometimes objects in env are not positioned on table but slightly above it.
        # Therefore, we do num_wait_steps simulation steps before we start the actual policy.


        # self.grasp_pos = grasp_pos
        # self.target_pos = target_pos
        # self.log = log
        # self.log_level = log_level

    # def _log(self, *msgs, log_level=0):
    #     if self.log and log_level <= self.log_level:
    #         print(msgs)

    # def get_action(self, obs):

    #     curr_hand_pos = obs[:3]

    #     correction = np.array([0.0, 0.0, -0.01])
    #     grasp_pos = self.grasp_pos + correction
    #     goal_pos = self.target_pos + correction

    #     action = np.zeros(4)
    #     self._log(self.curr_grasp_step, np.linalg.norm(grasp_pos - curr_hand_pos), np.linalg.norm(goal_pos - curr_hand_pos), log_level=1)

    #     top_pos_offset = np.array([0.,0.,0.05])
    #     err_threshold = 0.01

    #     to_xyz = curr_hand_pos
    #     if self.step == -1:
    #         # wait
    #         self.curr_wait_step += 1
    #         if self.curr_wait_step == self.num_wait_steps:
    #             self.step = 0
    #             self._log("proceed with step 0")
    #     if self.step == 0:
    #         # move above grasp position
    #         obj_top_pos = grasp_pos + top_pos_offset
    #         if np.linalg.norm(obj_top_pos - curr_hand_pos) < err_threshold:
    #             self.step = 1
    #             self._log("proceed with step 1")
    #         to_xyz = obj_top_pos # - curr_hand_pos
    #     elif self.step == 1:
    #         # move to grasp position
    #         if np.linalg.norm(grasp_pos - curr_hand_pos) < err_threshold:
    #             self.step = 2
    #             self._log("proceed with step 2")
    #         to_xyz = grasp_pos
    #     elif self.step == 2:
    #         # grasp
    #         action[3] = 1
    #         self.curr_grasp_step += 1
    #         if self.curr_grasp_step == self.grasp_steps:
    #             self.step = 3
    #             self._log("proceed with step 3")
    #     elif self.step == 3:
    #         # move above grasp position
    #         action[3] = 1
    #         obj_top_pos = grasp_pos + top_pos_offset
    #         if goal_pos[2] > grasp_pos[2]:
    #             obj_top_pos[2] = goal_pos[2] + top_pos_offset[2]
    #         if np.linalg.norm(obj_top_pos - curr_hand_pos) < err_threshold:
    #             self.step = 4
    #             self._log("proceed with step 4")
    #         to_xyz = obj_top_pos
    #     elif self.step == 4:
    #         # move above goal position
    #         action[3] = 1
    #         obj_top_pos = goal_pos + top_pos_offset
    #         if np.linalg.norm(obj_top_pos - curr_hand_pos) < err_threshold:
    #             self.step = 5
    #             self._log("proceed with step 5")
    #         to_xyz = obj_top_pos
    #     elif self.step == 5:
    #         # move to goal position
    #         action[3] = 1
    #         if np.linalg.norm(goal_pos - curr_hand_pos) < err_threshold:
    #             self.step = 6
    #             self._log("proceed with step 6")
    #         to_xyz = goal_pos

    #     action[:3] = move(curr_hand_pos, to_xyz=to_xyz, p=10.0)
    #     # action[:3] = action[:3] * 2 # / np.linalg.norm(action) NOTE if we devide by norm, velocity does not go down as we reach the goal
    #     # if np.linalg.norm(action[:3]) < 0.5:
    #     #     action[:3] = action[:3] / (np.linalg.norm(action[:3]) + err_threshold) * 0.5
    #     return action
