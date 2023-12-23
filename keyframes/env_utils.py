import numpy as np

import pytest

import keyframes.env_renderer as env_renderer
import keyframes.reset_space as reset_space
import keyframes.policies.agent as agent
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
)

from metaworld import MT1
from metaworld.policies import (
    SawyerAssemblyV2Policy,
    SawyerBasketballV2Policy,
    SawyerBinPickingV2Policy,
    SawyerBoxCloseV2Policy,
    SawyerButtonPressTopdownV2Policy,
    SawyerButtonPressTopdownWallV2Policy,
    SawyerButtonPressV2Policy,
    SawyerButtonPressWallV2Policy,
    SawyerCoffeeButtonV2Policy,
    SawyerCoffeePullV2Policy,
    SawyerCoffeePushV2Policy,
    SawyerDialTurnV2Policy,
    SawyerDisassembleV2Policy,
    SawyerDoorCloseV2Policy,
    SawyerDoorLockV2Policy,
    SawyerDoorOpenV2Policy,
    SawyerDoorUnlockV2Policy,
    SawyerDrawerCloseV2Policy,
    SawyerDrawerOpenV2Policy,
    SawyerFaucetCloseV2Policy,
    SawyerFaucetOpenV2Policy,
    SawyerHammerV2Policy,
    SawyerHandInsertV2Policy,
    SawyerHandlePressSideV2Policy,
    SawyerHandlePressV2Policy,
    SawyerHandlePullSideV2Policy,
    SawyerHandlePullV2Policy,
    SawyerLeverPullV2Policy,
    SawyerPegInsertionSideV2Policy,
    SawyerPegUnplugSideV2Policy,
    SawyerPickOutOfHoleV2Policy,
    SawyerPickPlaceV2Policy,
    SawyerPickPlaceWallV2Policy,
    SawyerPlateSlideBackSideV2Policy,
    SawyerPlateSlideBackV2Policy,
    SawyerPlateSlideSideV2Policy,
    SawyerPlateSlideV2Policy,
    SawyerPushBackV2Policy,
    SawyerPushV2Policy,
    SawyerPushWallV2Policy,
    SawyerReachV2Policy,
    SawyerReachWallV2Policy,
    SawyerShelfPlaceV2Policy,
    SawyerSoccerV2Policy,
    SawyerStickPullV2Policy,
    SawyerStickPushV2Policy,
    SawyerSweepIntoV2Policy,
    SawyerSweepV2Policy,
    SawyerWindowCloseV2Policy,
    SawyerWindowOpenV2Policy,
)

policies = dict(
    {
        "assembly-v2": SawyerAssemblyV2Policy,
        "basketball-v2": SawyerBasketballV2Policy,
        "bin-picking-v2": SawyerBinPickingV2Policy,
        "box-close-v2": SawyerBoxCloseV2Policy,
        "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy,
        "button-press-topdown-wall-v2": SawyerButtonPressTopdownWallV2Policy,
        "button-press-v2": SawyerButtonPressV2Policy,
        "button-press-wall-v2": SawyerButtonPressWallV2Policy,
        "coffee-button-v2": SawyerCoffeeButtonV2Policy,
        "coffee-pull-v2": SawyerCoffeePullV2Policy,
        "coffee-push-v2": SawyerCoffeePushV2Policy,
        "dial-turn-v2": SawyerDialTurnV2Policy,
        "disassemble-v2": SawyerDisassembleV2Policy,
        "door-close-v2": SawyerDoorCloseV2Policy,
        "door-lock-v2": SawyerDoorLockV2Policy,
        "door-open-v2": SawyerDoorOpenV2Policy,
        "door-unlock-v2": SawyerDoorUnlockV2Policy,
        "drawer-close-v2": SawyerDrawerCloseV2Policy,
        "drawer-open-v2": SawyerDrawerOpenV2Policy,
        "faucet-close-v2": SawyerFaucetCloseV2Policy,
        "faucet-open-v2": SawyerFaucetOpenV2Policy,
        "hammer-v2": SawyerHammerV2Policy,
        "hand-insert-v2": SawyerHandInsertV2Policy,
        "handle-press-side-v2": SawyerHandlePressSideV2Policy,
        "handle-press-v2": SawyerHandlePressV2Policy,
        "handle-pull-v2": SawyerHandlePullV2Policy,
        "handle-pull-side-v2": SawyerHandlePullSideV2Policy,
        "peg-insert-side-v2": SawyerPegInsertionSideV2Policy,
        "lever-pull-v2": SawyerLeverPullV2Policy,
        "peg-unplug-side-v2": SawyerPegUnplugSideV2Policy,
        "pick-out-of-hole-v2": SawyerPickOutOfHoleV2Policy,
        "pick-place-v2": SawyerPickPlaceV2Policy,
        "pick-place-wall-v2": SawyerPickPlaceWallV2Policy,
        "plate-slide-back-side-v2": SawyerPlateSlideBackSideV2Policy,
        "plate-slide-back-v2": SawyerPlateSlideBackV2Policy,
        "plate-slide-side-v2": SawyerPlateSlideSideV2Policy,
        "plate-slide-v2": SawyerPlateSlideV2Policy,
        "reach-v2": SawyerReachV2Policy,
        "reach-wall-v2": SawyerReachWallV2Policy,
        "push-back-v2": SawyerPushBackV2Policy,
        "push-v2": SawyerPushV2Policy,
        "push-wall-v2": SawyerPushWallV2Policy,
        "shelf-place-v2": SawyerShelfPlaceV2Policy,
        "soccer-v2": SawyerSoccerV2Policy,
        "stick-pull-v2": SawyerStickPullV2Policy,
        "stick-push-v2": SawyerStickPushV2Policy,
        "sweep-into-v2": SawyerSweepIntoV2Policy,
        "sweep-v2": SawyerSweepV2Policy,
        "window-close-v2": SawyerWindowCloseV2Policy,
        "window-open-v2": SawyerWindowOpenV2Policy,
    }
)


@pytest.mark.parametrize("env_name", MT1.ENV_NAMES)
def test_policy(env_name):
    mt1 = MT1(env_name)
    env = mt1.train_classes[env_name]()
    p = policies[env_name]()
    completed = 0
    for task in mt1.train_tasks:
        env.set_task(task)
        obs, info = env.reset()
        done = False
        count = 0
        while count < 500 and not done:
            count += 1
            a = p.get_action(obs)
            next_obs, _, trunc, termn, info = env.step(a)
            done = trunc or termn
            obs = next_obs
            if int(info["success"]) == 1:
                completed += 1
                break
    print(float(completed) / 50)
    assert (float(completed) / 50) > 0.80

# env_configs = [
#     # env, action noise pct, cycles, quit on success
#     ('assembly-v2', np.zeros(4), 1, True),
#     # ('basketball-v2', np.zeros(4), 3, True),
#     ('bin-picking-v2', np.zeros(4), 3, True),
#     ('box-close-v2', np.zeros(4), 3, True),
#     ('button-press-topdown-v2', np.zeros(4), 3, True),
#     ('button-press-topdown-wall-v2', np.zeros(4), 3, True),
#     ('button-press-v2', np.zeros(4), 3, True),
#     ('button-press-wall-v2', np.zeros(4), 3, True),
#     ('coffee-button-v2', np.zeros(4), 3, True),
#     ('coffee-pull-v2', np.zeros(4), 3, True),
#     ('coffee-push-v2', np.zeros(4), 3, True),
#     ('dial-turn-v2', np.zeros(4), 3, True),
#     ('disassemble-v2', np.zeros(4), 3, True),
#     ('door-close-v2', np.zeros(4), 3, True),
#     ('door-lock-v2', np.zeros(4), 3, True),
#     ('door-open-v2', np.zeros(4), 3, True),
#     ('door-unlock-v2', np.zeros(4), 3, True),
#     ('hand-insert-v2', np.zeros(4), 3, True),
#     ('drawer-close-v2', np.zeros(4), 3, True),
#     ('drawer-open-v2', np.zeros(4), 3, True),
#     ('faucet-open-v2', np.zeros(4), 3, True),
#     ('faucet-close-v2', np.zeros(4), 3, True),
#     ('hammer-v2', np.zeros(4), 3, True),
#     ('handle-press-side-v2', np.zeros(4), 3, True),
#     ('handle-press-v2', np.zeros(4), 3, True),
#     ('handle-pull-side-v2', np.zeros(4), 3, True),
#     ('handle-pull-v2', np.zeros(4), 3, True),
#     ('lever-pull-v2', np.zeros(4), 3, True),
#     ('peg-insert-side-v2', np.zeros(4), 3, True),
#     ('pick-place-wall-v2', np.zeros(4), 3, True),
#     ('pick-out-of-hole-v2', np.zeros(4), 3, True),
#     ('reach-v2', np.zeros(4), 3, True),
#     ('push-back-v2', np.zeros(4), 3, True),
#     ('push-v2', np.zeros(4), 3, True),
#     ('pick-place-v2', np.zeros(4), 3, True),
#     ('plate-slide-v2', np.zeros(4), 3, True),
#     ('plate-slide-side-v2', np.zeros(4), 3, True),
#     ('plate-slide-back-v2', np.zeros(4), 3, True),
#     ('plate-slide-back-side-v2', np.zeros(4), 3, True),
#     ('peg-unplug-side-v2', np.zeros(4), 3, True),
#     ('soccer-v2', np.zeros(4), 3, True),
#     ('stick-push-v2', np.zeros(4), 3, True),
#     ('stick-pull-v2', np.zeros(4), 3, True),
#     ('push-wall-v2', np.zeros(4), 3, True),
#     ('reach-wall-v2', np.zeros(4), 3, True),
#     ('shelf-place-v2', np.zeros(4), 3, True),
#     ('sweep-into-v2', np.zeros(4), 3, True),
#     ('sweep-v2', np.zeros(4), 3, True),
#     ('window-open-v2', np.zeros(4), 3, True),
#     ('window-close-v2', np.zeros(4), 3, True),
# ]

env_names = MT1.ENV_NAMES

def generate_env_and_renderer(env_index = 0, height=240, width: int = 320, camera_name="keyframes"):
    """
    Args:
        NOTE: see metaworld/envs/assets_v2/objects/assets/xyz_base.xml
            <camera pos="0 0.5 1.5" name="topview"/>
            <!-- <camera name="keyframes" mode="fixed" pos="0 0.0 1.5"  euler="0.6 0 0"/> -->
            
            <camera name="keyframes" mode="fixed" pos="0 0.0 1.5" quat="1 0.17 0 0"/>
            <camera name="corner" mode="fixed" pos="-1.1 -0.4 0.6" xyaxes="-1 1 0 -0.2 -0.2 -1"/>
            <camera name="corner2" fovy="60" mode="fixed" pos="1.3 -0.2 1.1" euler="3.9 2.3 0.6"/>
            <camera name="corner3" fovy="45" mode="fixed" pos="0.9 0 1.5" euler="3.5 2.7 1"/>
        camera_name: topview, keyframes, corner, corner2, corner3
    """


    # setup env
    env_name = env_names[env_index]
    env_scripted_policy = policies[env_name]
    mt1 = MT1(env_name)
    env = mt1.train_classes[env_name]()

    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env._random_reset_space = reset_space.get_custom_random_reset_space(env_name)
    env.hand_init_pos = reset_space.get_random_hand_init_pos()

    env.reset_model()
    env.reset()

    # renderer
    # NOTE keyframes camera defined in metaworld/envs/assets_v2/objects/assets/xyz_base.xml
    renderer = env_renderer.EnvRenderer(env, camera_name=camera_name, height=height, width=width)
    # env.model.camera("keyframes").quat = [1,0.17,0,0]

    return env, env_scripted_policy, renderer


def go_forward(
    env: SawyerXYZEnv,
    renderer: env_renderer.EnvRenderer = None,
    agent: agent.Agent = None,
    n_steps=10,
):
    imgs = []
    o = env._prev_obs
    for i in range(n_steps):
        if agent is None:
            a = np.array([0, 0, 0, 0])
        else:
            a = agent.get_action(o)
        # a = np.array([0,0.1,0,0])
        # print("action", a, o[:3])
        o, r, _, done, info = env.step(a)
        if done or info["success"] > 0:
            print("done", info["success"] > 0, i)
            break
        if renderer is not None:
            imgs.append(renderer.render()["img"].copy())
    records = {
        "imgs": imgs,
        "o": o,
        "r": r,
        "done": done,
        "info": info,
    }
    return records
