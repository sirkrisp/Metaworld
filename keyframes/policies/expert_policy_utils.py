
import numpy as np
import keyframes.env_utils as env_utils
import keyframes.env_renderer as env_renderer

import keyframes.policies.agent as agent
import keyframes.policies.pick_place_policy as pick_place_policy
import keyframes.policies.push_policy as push_policy

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
)



def get_policy(env, env_name, o):
    metaworld_policy = env_utils.policies[env_name]
    o_d = metaworld_policy._parse_obs(o)
    if env_name == "assembly-v2":
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])
        pos_peg = o_d["peg_pos"] + np.array([0.12, 0.0, 0.14])
        # + np.array([0.0, 0.0, 0.1])
        grasp_pos = pos_wrench + np.array([0.0, 0.0, 0.05])
        # target_pos = pos_peg + np.array([0.0, 0.0, -0.1])
        target_pos = env._target_pos + np.array([0.1, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "basketball-v2":
        pos_ball = o_d["ball_pos"] + np.array([0.0, 0.0, 0.05])
        # X is given by hoop_pos
        # Y varies between .85 and .9, so we take avg
        # Z is constant at .35
        # pos_hoop = np.array([o_d["hoop_x"], *o_d["hoop_yz"]]) + np.array([-0.0, -0.03, 0.05])
        target_pos = env._target_pos + np.array([0.0, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(pos_ball, target_pos)
    if env_name == "bin-picking-v2":
        pass
    if env_name == "box-close-v2":
        # Problem with env => box cap is not in the right position
        pass
    if env_name == "button-press-topdown-v2":
        push_start_pos = o_d["button_pos"] + np.array([0.0, 0.0, 0.05])
        target_pos = push_start_pos + np.array([0.0, 0.0, -0.2])
        return push_policy.PushPolicy(push_start_pos, target_pos)
    if env_name == "button-press-topdown-wall-v2":
        push_start_pos = o_d["button_pos"] + np.array([0.0, 0.0, 0.05])
        target_pos = push_start_pos + np.array([0.0, 0.0, -0.2])
        return push_policy.PushPolicy(push_start_pos, target_pos)
    if env_name == "button-press-v2":
        push_start_pos = o_d["button_pos"] + np.array([0.0, -0.05, 0.0])
        target_pos = push_start_pos + np.array([0.0, 0.1, 0.0])
        return push_policy.PushPolicy(push_start_pos, target_pos)
    if env_name == "button-press-wall-v2":
        push_start_pos = o_d["button_pos"] + np.array([0.0, -0.05, 0.0])
        target_pos = push_start_pos + np.array([0.0, 0.1, 0.0])
        return push_policy.PushPolicy(push_start_pos, target_pos)
        # NOTE does not work due to collision
        pass
    if env_name == "coffee-button-v2":
        push_start_pos = o_d["button_pos"] + np.array([0.0, 0.0, -0.07])
        target_pos = o_d["button_pos"] + np.array([0.0, 0.0, 0.2])
        return push_policy.PushPolicy(push_start_pos, target_pos)
    if env_name == "coffee-pull-v2":
        grasp_pos = o_d["mug_pos"] + np.array([-0.005, 0.0, 0.1])
        # TODO why is target_pos not the same as in the environment?
        # target_pos = o_d["target_pos"] + np.array([0.0, 0.0, 0.05])
        target_pos = env._target_pos + np.array([0.0, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "coffee-push-v2":
        grasp_pos = o_d["mug_pos"] + np.array([-0.005, 0.0, 0.1])
        target_pos = env._target_pos + np.array([0.0, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "dial-turn-v2":
        pass
    if env_name == "disassemble-v2":
        # NOTE this should be pull instead of pick and place
        pos_wrench = o_d["wrench_pos"] + np.array([-0.02, 0.0, 0.0])
        pos_peg = o_d["peg_pos"] + np.array([0.12, 0.0, 0.14])
        # + np.array([0.0, 0.0, 0.1])
        grasp_pos = pos_wrench + np.array([0.0, 0.0, 0.05])
        # target_pos = pos_peg + np.array([0.0, 0.0, -0.1])
        target_pos = env._target_pos + np.array([0.1, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "door-close-v2":
        pass
    if env_name == "door-lock-v2":
        pass
    if env_name == "door-open-v2":
        pass
    if env_name == "door-unlock-v2":
        pass
    if env_name == "hand-insert-v2":
        grasp_pos = o_d["obj_pos"] + np.array([0.0, 0.0, 0.05])
        target_pos = o_d["goal_pos"] + np.array([0.0, 0.0, 0.05])
        # NOTE works much better if target_pos z is greater than 0.05
        # target_pos[2] = np.max([target_pos[2], 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "drawer-close-v2":
        push_start_pos = o_d["drwr_pos"] + np.array([0.0, -0.1, 0.00])
        target_pos = push_start_pos + np.array([0.0, 0.2, 0.0])
        return push_policy.PushPolicy(push_start_pos, target_pos)
    if env_name == "drawer-open-v2":
        push_start_pos = o_d["drwr_pos"] + np.array([0.0, -0.05, 0.00])
        target_pos = push_start_pos + np.array([0.0, -0.1, 0.0])
        return push_policy.PushPolicy(push_start_pos, target_pos)
    if env_name == "faucet-open-v2":
        pass
    if env_name == "faucet-close-v2":
        pass
    if env_name == "hammer-v2":
        pass
    if env_name == "handle-press-side-v2":
        push_start_pos = o_d["handle_pos"] + np.array([0.0, 0.0, 0.2])
        target_pos = o_d["handle_pos"] + np.array([0.0, 0.0, -0.5])
        return push_policy.PushPolicy(push_start_pos, target_pos)
    if env_name == "handle-press-v2":
        push_start_pos = o_d["handle_pos"] + np.array([0.0, 0.0, 0.2])
        target_pos = o_d["handle_pos"] + np.array([0.0, 0.0, -0.5])
        return push_policy.PushPolicy(push_start_pos, target_pos)
    if env_name == "handle-pull-side-v2":
        pass
    if env_name == "handle-pull-v2":
        pass
    if env_name == "lever-pull-v2":
        pass
    if env_name == "peg-insert-side-v2":
        pass
    if env_name == "pick-place-wall-v2":
        pass
    if env_name == "pick-out-of-hole-v2":
        grasp_pos = o_d["puck_pos"] + np.array([-0.01, 0.0, 0.04])
        target_pos = o_d["goal_pos"] + np.array([0.0, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "reach-v2":
        # NOTE not a manipulation task
        pass
    if env_name == "push-back-v2":
        grasp_pos = o_d["puck_pos"] + np.array([0.0, 0.0, 0.05])
        target_pos = o_d["goal_pos"] + np.array([0.0, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "push-v2":
        grasp_pos = o_d["puck_pos"] + np.array([-0.01, 0.0, 0.05])
        # target_pos = o_d["goal_pos"] + np.array([0.0, 0.0, 0.05])
        target_pos = env._target_pos + np.array([0.0, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "pick-place-v2":
        grasp_pos = o_d["puck_pos"] + np.array([-0.0, 0.0, 0.05])
        # target_pos = o_d["goal_pos"] + np.array([0.0, 0.0, 0.05])
        target_pos = env._target_pos + np.array([0.0, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "plate-slide-v2":
        pass
    if env_name == "plate-slide-side-v2":
        pass
    if env_name == "plate-slide-back-v2":
        pass
    if env_name == "plate-slide-back-side-v2":
        pass
    if env_name == "peg-unplug-side-v2":
        pass
    if env_name == "soccer-v2":
        grasp_pos = o_d["ball_pos"] + np.array([0.0, 0.0, 0.05])
        target_pos = o_d["goal_pos"] + np.array([0.0, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "stick-push-v2":
        pass
    if env_name == "stick-pull-v2":
        pass
    if env_name == "push-wall-v2":
        pass
    if env_name == "reach-wall-v2":
        pass
    if env_name == "shelf-place-v2":
        pos_curr = o_d["hand_pos"]
        grasp_pos = o_d["block_pos"] + np.array([-0.005, 0.0, 0.05])
        target_pos = np.array([o_d["shelf_x"], *o_d["unused_3"]]) + np.array(
            [0.0, 0.0, 0.05]
        )
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "sweep-into-v2":
        grasp_pos = o_d["cube_pos"] + np.array([-0.005, 0.0, 0.05])
        target_pos = o_d["goal_pos"] + np.array([0.0, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "sweep-v2":
        grasp_pos = o_d["cube_pos"] + np.array([-0.005, 0.0, 0.05])
        target_pos = o_d["goal_pos"] + np.array([0.0, 0.0, 0.05])
        return pick_place_policy.PickPlacePolicy(grasp_pos, target_pos)
    if env_name == "window-open-v2":
        pos_wndw = o_d["wndw_pos"] + np.array([-0.03, -0.01, -0.08])
        push_start_pos = pos_wndw + np.array([-0.05, 0.0, 0.1])
        target_pos = pos_wndw + np.array([0.2, 0.0, 0.1])
        return push_policy.PushPolicy(push_start_pos, target_pos)
    if env_name == "window-close-v2":
        pos_wndw = o_d["wndw_pos"] + np.array([-0.03, -0.0, -0.08])
        push_start_pos = pos_wndw + np.array([+0.05, -0.0, 0.15])
        target_pos = pos_wndw + np.array([-0.4, -0.03, 0.15])
        return push_policy.PushPolicy(push_start_pos, target_pos)
