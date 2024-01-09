import keyframes.env_renderer as env_renderer
import keyframes.mujoco_utils as mujoco_utils
import numpy as np
import mujoco
import utils.slam_utils as slam_utils
import keyframes.env_utils as env_utils

def generate_keyframes(env, renderer):
    # reset env
    env.reset()
    # TODO why needed?
    mujoco.mj_resetData(env.model, env.data)

    # do not visualize robot bodies and sites
    root_node, node_map = mujoco_utils.build_mj_tree(env.model)
    name2node = {}
    for node in node_map.values():
        name2node[node.name] = node
    mujoco_utils.toggle_sites_visibility(env.model, False)
    siteSet = set(mujoco_utils.get_site_names(env.model))
    # TODO only visualize goal if env.show_goal is True
    # if "goal" in siteSet:
    #     env.model.site("goal").rgba = [0,0,1,1]
    mujoco_utils.toggle_visibility(env.model, name2node["base"], False)

    # generate keyframes
    render_keys = ["img", "depth", "seg"]
    keyframes_data = {
        "img": [],
        "depth": [],
        "seg": [],
        "geom_xpos": [], # geom positions
        "geom_xmat": [], # geom rotations
    }
    for i in range(env.get_num_steps() + 1):
        env.go_to_step(i)
        mujoco.mj_forward(env.model, env.data)
        render_data = renderer.render(depth=True, segmentation=True)
         
        for k in render_keys:
            keyframes_data[k].append(render_data[k])

        # geom positions and rotations
        keyframes_data["geom_xpos"].append(env.data.geom_xpos.copy())
        keyframes_data["geom_xmat"].append(env.data.geom_xmat.copy())
        
    for k in keyframes_data.keys():
        keyframes_data[k] = np.array(keyframes_data[k])

    # TODO what about sites?
    mujoco_utils.toggle_visibility(env.model, name2node["base"], True)

    return keyframes_data


def generate_cur_keyframe(env, renderer):
    # reset env
    env.reset()
    # TODO why needed?
    mujoco.mj_resetData(env.model, env.data)

    # do not visualize robot bodies and sites
    root_node, node_map = mujoco_utils.build_mj_tree(env.model)
    name2node = {}
    for node in node_map.values():
        name2node[node.name] = node
    mujoco_utils.toggle_sites_visibility(env.model, False)
    siteSet = set(mujoco_utils.get_site_names(env.model))
    # TODO only visualize goal if env.show_goal is True
    # if "goal" in siteSet:
    #     env.model.site("goal").rgba = [0,0,1,1]
    mujoco_utils.toggle_visibility(env.model, name2node["base"], False)

    env.go_to_step(0)
    mujoco.mj_forward(env.model, env.data)
    o = env_utils.go_forward(env, n_steps=100)["o"]
    
    render_data = renderer.render(depth=True, segmentation=False)
    img = render_data["img"].copy()
    depth = render_data["depth"].copy()

    mujoco_utils.toggle_visibility(env.model, name2node["base"], True)

    return img, depth



def generate_keyframe_data_with_features(env, renderer, extractor):
    keyframe_data = generate_keyframes(env, renderer)
    keypoints = []
    keypoint_scores = []
    descriptors = []
    for i in range(2):
        fts = slam_utils.compute_features(extractor, keyframe_data["img"][i])
        keypoints.append(fts["keypoints"].cpu().numpy())
        keypoint_scores.append(fts["keypoint_scores"].cpu().numpy())
        descriptors.append(fts["descriptors"].cpu().numpy())
    keyframe_data["keypoints"] = keypoints
    keyframe_data["keypoint_scores"] = keypoint_scores
    keyframe_data["descriptors"] = descriptors
    return keyframe_data

