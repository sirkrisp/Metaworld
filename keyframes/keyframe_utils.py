import env_renderer
import mujoco_utils
import numpy as np
import mujoco

def generate_keyframes(env, renderer):
    # reset env
    env.reset()

    # do not visualize robot bodies and sites
    root_node, node_map = mujoco_utils.build_mj_tree(env.model)
    name2node = {}
    for node in node_map.values():
        name2node[node.name] = node
    mujoco_utils.toggle_sites_visibility(env.model, False)
    siteSet = set(mujoco_utils.get_site_names(env.model))
    # TODO only visualize goal if env.show_goal is True
    if "goal" in siteSet:
        env.model.site("goal").rgba = [0,0,1,1]
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
        keyframes_data["geom_xpos"].append(env.data.geom_xpos[:])
        keyframes_data["geom_xmat"].append(env.data.geom_xmat[:])
        
    for k in keyframes_data.keys():
        keyframes_data[k] = np.array(keyframes_data[k])

    return keyframes_data



