from gymnasium.spaces import Box
import numpy as np

def sample_hand_pos_around_object(obj_pos):
    low = obj_pos - np.array([0.2, 0.2, 0.1])
    high = obj_pos + np.array([0.2, 0.2, 0.3])
    point = low + np.random.uniform(0, 1, 3) * (high - low)
    # if hand is too close to object, sample again
    while np.linalg.norm(point - obj_pos) < 0.1:
        point = low + np.random.uniform(0, 1, 3) * (high - low)
    return point

def get_random_hand_init_pos(low = np.array([-0.5, 0.3, 0.1]), high = np.array([0.5, 1.0, 0.5])):
    return low + np.random.uniform(0, 1, 3) * (high - low)

def get_custom_random_reset_space(task):
    """
    Adapt the random reset space to the task.

    X, Y, Z centered at the robot base and aligned with robot.
    """
    obj_low, obj_high, goal_low, goal_high = None, None, None, None
    if task == 'assembly-v2':
        obj_low = (-0.4, 0.4, 0.02)
        obj_high = (0.4, 0.65, 0.02)
        goal_low = (-0.3, 0.75, 0.1)
        goal_high = (0.3, 0.85, 0.1)
    elif task == 'basketball-v2':
        obj_low = (-0.4, 0.4, 0.0299)
        obj_high = (0.4, 0.7, 0.0301)
        goal_low = (-0.2, 0.8, 0.)
        goal_high = (0.2, 0.9+1e-7, 0.)
    elif task == 'bin-picking-v2':
        # Space is constrined by bins
        obj_low = (-0.17, 0.65, 0.02)
        obj_high = (-0.07, 0.75, 0.02)
        goal_low = (0.1199, 0.8, -0.001)
        goal_high = (0.1201, 0.8, +0.001)
    elif task == 'box-close-v2':
        # TODO Z coordinate seems to have no impact
        obj_low = (-0.3, 0.45, 0.02)
        obj_high = (0.3, 0.5, 0.02)
        goal_low = (-0.2, 0.7, 0.133)
        goal_high = (0.2, 0.8, 0.133)
    elif task == 'button-press-topdown-v2':
        obj_low = (-0.4, 0.7, 0.115)
        obj_high = (0.4, 0.85, 0.115)
    elif task == 'button-press-topdown-wall-v2':
        obj_low = (-0.3, 0.8, 0.115)
        obj_high = (0.3, 0.9, 0.115)
    elif task == 'button-press-v2':
        obj_low = (-0.3, 0.75, 0.115)
        obj_high = (0.3, 0.9, 0.115)
    elif task == 'button-press-wall-v2':
        obj_low = (-0.1, 0.95, 0.1149)
        obj_high = (0.3, 1, 0.1151)
    elif task == 'coffee-button-v2':
        obj_low = (-0.3, 0.8, -.001)
        obj_high = (0.3, 0.9, 0.3)
    elif task == 'coffee-pull-v2':
        obj_low = (-0.3, 0.7, -.001)
        obj_high = (0.3, 0.8, +.001)
        goal_low = (-0.3, 0.4, -.001)
        goal_high = (0.3, 0.65, +.001)
    elif task == 'coffee-push-v2':
        obj_low = (-0.4, 0.45, -.001)
        obj_high = (0.4, 0.55, +.001)
        goal_low = (-0.25, 0.7, -.001)
        goal_high = (0.25, 0.8, +.001)
    elif task == 'dial-turn-v2':
        obj_low = (-0.4, 0.6, 0.0)
        obj_high = (0.4, 0.8, 0.0)
    elif task == 'disassemble-v2':
        obj_low = (-0.25, 0.55, 0.025)
        obj_high = (0.25, 0.65, 0.02501)
        goal_low = (-0.25, 0.55, 0.1699)
        goal_high = (0.25, 0.65, 0.1701)
    elif task == 'door-close-v2':
        obj_low = (0., 0.85, 0.15)
        obj_high = (0.4, 0.95, 0.15)
    elif task == 'door-lock-v2':
        obj_low = (-0.3, 0.8, 0.15)
        obj_high = (0.3, 0.85, 0.15)
    elif task == 'door-open-v2':
        obj_low = (0., 0.85, 0.15)
        obj_high = (0.3, 0.95, 0.15)
    elif task == 'door-unlock-v2':
        obj_low = (0.0, 0.8, 0.15)
        obj_high = (0.4, 0.85, 0.15)
    elif task == 'hand-insert-v2':
        obj_low = (-0.4, 0.45, 0.05)
        obj_high = (0.4, 0.7, 0.05)
        goal_low = (-0.04, 0.8, -0.0201)
        goal_high = (0.04, 0.88, -0.0199)
    elif task == 'drawer-close-v2':
        obj_low = (-0.3, 0.9, 0.)
        obj_high = (0.3, 0.9, 0.)
    elif task == 'drawer-open-v2':
        obj_low = (-0.3, 0.9, 0.)
        obj_high = (0.3, 0.9, 0.)
    elif task == 'faucet-open-v2':
        obj_low = (-0.2, 0.7, 0.)
        obj_high = (0.3, 0.85, 0.)
    elif task == 'faucet-close-v2':
        obj_low = (-0.3, 0.7, 0.)
        obj_high = (0.2, 0.85, 0.)
    elif task == 'hammer-v2':
        obj_low = (-0.3, 0.4, 0.)
        obj_high = (0.3, 0.5, 0.)
    elif task == 'handle-press-side-v2':
        obj_low = (-0.35, 0.8, -0.001)
        obj_high = (0.2, 0.9, 0.001)
    elif task == 'handle-press-v2':
        obj_low = (-0.4, 0.8, -0.001)
        obj_high = (0.4, 0.9, 0.001)
    elif task == 'handle-pull-side-v2':
        # TODO policy not working for this space
        obj_low = (-0.35, 0.8, -0.001)
        obj_high = (0.2, 0.9, 0.001)
    elif task == 'handle-pull-v2':
        # NOTE weird policy
        obj_low = (-0.4, 0.8, -0.001)
        obj_high = (0.4, 0.9, 0.001)
    elif task == 'lever-pull-v2':
        obj_low = (-0.1, 0.7, 0.)
        obj_high = (0.1, 0.8, 0.)
    elif task == 'peg-insert-side-v2':
        obj_low = (0.15, 0.4, 0.02)
        obj_high = (0.3, 0.7, 0.02)
        goal_low = (-0.3, 0.4, -0.001)
        goal_high = (-0.15, 0.7, 0.001)
    elif task == 'pick-place-wall-v2':
        obj_low = (-0.2, 0.6, 0.015)
        obj_high = (0.2, 0.65, 0.015)
        goal_low = (-0.2, 0.85, 0.05)
        goal_high = (0.2, 0.9, 0.3)
    elif task == 'pick-out-of-hole-v2':
        # NOTE weird policy
        obj_low = (-0.1, 0.75, 0.02)
        obj_high = (0.1, 0.85, 0.02)
        goal_low = (-0.3, 0.5, 0.15)
        goal_high = (0.3, 0.6, 0.3)
    elif task == 'reach-v2':
        # NOTE weird environment (we do not pick up the object) => see pick-place-v2
        obj_low = (-0.4, 0.4, 0.02)
        obj_high = (0.4, 0.7, 0.02)
        goal_low = (-0.4, 0.7, 0.05)
        goal_high = (0.4, 0.9, 0.3)
    elif task == 'push-back-v2':
        obj_low = (-0.25, 0.8, 0.02)
        obj_high = (0.25, 0.85, 0.02)
        goal_low = (-0.25, 0.6, 0.0199)
        goal_high = (0.25, 0.7, 0.0201)
    elif task == 'push-v2':
        obj_low = (-0.25, 0.6, 0.02)
        obj_high = (0.25, 0.7, 0.02)
        goal_low = (-0.25, 0.8, 0.01)
        goal_high = (0.25, 0.9, 0.02)
    elif task == 'pick-place-v2':
        obj_low = (-0.4, 0.4, 0.02)
        obj_high = (0.4, 0.7, 0.02)
        goal_low = (-0.4, 0.7, 0.05)
        goal_high = (0.4, 0.9, 0.3)
    elif task == 'plate-slide-v2':
        obj_low = (-0.25, 0.4, 0.)
        obj_high = (0.25, 0.6, 0.)
        goal_low = (-0.2, 0.85, 0.)
        goal_high = (0.2, 0.9, 0.)
    elif task == 'plate-slide-side-v2':
        # NOTE policy does not account for y offset
        obj_low = (0., 0.6, 0.)
        obj_high = (0.3, 0.6, 0.)
        goal_low = (-0.4, 0.54, 0.)
        goal_high = (-0.2, 0.66, 0.)
    elif task == 'plate-slide-back-v2':
        # NOTE polic is not robust, position of disk cannot be changed
        obj_low = (-0.1, 0.85, 0.)
        obj_high = (0.1, 0.85, 0.)
        goal_low = (-0.1, 0.6, 0.015)
        goal_high = (0.1, 0.6, 0.015)
    elif task == 'plate-slide-back-side-v2':
        # NOTE values not changed
        obj_low = (-0.25, 0.5, 0.)
        obj_high = (-0.25, 0.7, 0.)
        goal_low = (-0.05, 0.5, 0.015)
        goal_high = (0.15, 0.7, 0.015)
    elif task == 'peg-unplug-side-v2':
        obj_low = (-0.4, 0.4, -0.001)
        obj_high = (-0.15, 0.8, 0.001)
    elif task == 'soccer-v2':
        # NOTE policy is not very robust
        obj_low = (-0.1, 0.5, 0.03)
        obj_high = (0.1, 0.7, 0.03)
        goal_low = (-0.1, 0.8, 0.)
        goal_high = (0.1, 0.9, 0.)
    elif task == 'stick-push-v2':
        obj_low = (-0.3, 0.4, 0.)
        obj_high = (-0.03, 0.8, 0.001)
        # NOTE we cannot modify the position of the vase, so we cannot change the goal space too much
        goal_low = (0.399, 0.55, 0.1319)
        goal_high = (0.42, 0.6, 0.1321)
    elif task == 'stick-pull-v2':
        obj_low = (-0.3, 0.45, 0.)
        obj_high = (0., 0.75, 0.001)
        goal_low = (0.35, 0.45, 0.0199)
        goal_high = (0.45, 0.55, 0.0201)
    elif task == 'push-wall-v2':
        obj_low = (-0.3, 0.5, 0.015)
        obj_high = (0.05, 0.65, 0.015)
        goal_low = (-0.1, 0.85, 0.01)
        goal_high = (0.05, 0.9, 0.02)
    elif task == 'reach-wall-v2':
        obj_low = (-0.3, 0.5, 0.015)
        obj_high = (0.3, 0.65, 0.015)
        goal_low = (-0.1, 0.85, 0.05)
        goal_high = (0.05, 0.9, 0.3)
    elif task == 'shelf-place-v2':
        obj_low = (-0.3, 0.45, 0.019)
        obj_high = (0.3, 0.6, 0.021)
        goal_low = (-0.1, 0.8, 0.299)
        goal_high = (0.1, 0.9, 0.301)
    elif task == 'sweep-into-v2':
        obj_low = (-0.3, 0.6, 0.02)
        obj_high = (0.3, 0.7, 0.02)
        goal_low = (-0.001, 0.8399, 0.0199)
        goal_high = (0.001, 0.8401, 0.0201)
    elif task == 'sweep-v2':
        obj_low = (-0.3, 0.6, 0.02)
        obj_high = (0.1, 0.8, 0.02)
    elif task == 'window-open-v2':
        obj_low = (-0.3, 0.7, 0.16)
        obj_high = (0.3, 0.9, 0.16)
    elif task == 'window-close-v2':
        obj_low = (-0.3, 0.7, 0.2)
        obj_high = (0.2, 0.9, 0.2)
    if goal_high and goal_low and obj_high and obj_low:
        return Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
    elif obj_high and obj_low:
        return Box(
            np.array(obj_low),
            np.array(obj_high),
        )
    raise NotImplementedError
