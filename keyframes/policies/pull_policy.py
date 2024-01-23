import numpy as np
import keyframes.policies.agent as agent
import keyframes.policies.policy_path_points as policy_path_points


class PullPolicy(agent.Agent):
    """
    This class implements a pull policy.
    """

    def __init__(
        self,
        pull_start_pos: np.ndarray,
        pull_goal_pos: np.ndarray,
        log=False,
        log_level=0,
    ):
        pull_dir = pull_goal_pos - pull_start_pos
        pull_axis = int(np.argmax(np.abs(pull_dir)))

        # we grasp when pull in x or z and push when pull in y
        # (due to fixed orientation of gripper)
        open_gripper = pull_axis == 0 or pull_axis == 2 or pull_axis == 1
        initial_gripper_action = -1.0 if open_gripper else 1.0

        # pull_offset = np.array([0,0,-0.6]) if open_gripper else np.array([0,0.025,0])

        above_pull_start_pos = pull_start_pos + np.array([0.0, 0.0, min(pull_start_pos[2] + 0.3, 0.4)])
        move_up_z = above_pull_start_pos[2]

        threshold = 0.01
        p = 10.0

        path_points = [
            # Wait
            policy_path_points.Wait(num_wait_steps=40),
            # Move up (or down depending on current ee position)
            policy_path_points.MoveAxis(
                axis=2,
                value=move_up_z,
                gripper_action=initial_gripper_action,
                p=p,
                threshold=threshold,
                max_steps=10,
            ),
            # Move above pull start position
            policy_path_points.MoveTo(
                pos=above_pull_start_pos,
                gripper_action=initial_gripper_action,
                p=p,
                threshold=threshold,
                max_steps=100
            ),
            # Move to pull start position
            policy_path_points.MoveTo(
                pos=pull_start_pos, gripper_action=initial_gripper_action, p=p, threshold=threshold, max_steps=100
            ),
            # Grasp
            policy_path_points.Gripper(gripper_action=0.6, num_gripper_steps=30),
            policy_path_points.MoveTo(
                pos=pull_goal_pos, gripper_action=0.6, p=1, threshold=threshold, max_steps=100
            ),
            # policy_path_points.PeriodicPathPoint(
            #     [
            #         policy_path_points.Gripper(gripper_action=1.0, num_gripper_steps=10),
            #         # Move to pull goal position
            #         policy_path_points.MoveAxisDiff(
            #             axis=pull_axis, d_axis=0.1, gripper_action=1.0, p=10, threshold=threshold, max_steps=30
            #         ),
            #         policy_path_points.Gripper(gripper_action=-1.0, num_gripper_steps=10),
            #         policy_path_points.MoveAxisDiff(
            #             axis=pull_axis, d_axis=-0.05, gripper_action=1.0, p=10, threshold=threshold, max_steps=10
            #         )
            #     ],
            #     max_periods=40
            # )
        ]

        super().__init__(path_points=path_points, log=log, log_level=log_level)