ROOT_DIR = "/home/user/Documents/projects/Metaworld"

import sys
sys.path.append(ROOT_DIR)

import os
os.environ["MUJOCO_GL"] = "egl"

from tqdm import tqdm
import numpy as np
import torch

# project imports
import utils.camera_utils_v2 as cu
import utils.slam_utils as slam_utils
import keyframes.env_utils as env_utils
import keyframes.keyframe_utils as keyframe_utils


# ==============================================
#               PARAMETERS
# ==============================================

NUM_SAMPLES_PER_ENV = 200
EXCLUDED_ENV_IDS = [1,22,31,41,42]
DEVICE = "cuda"
IMG_HEIGHT = 360  # 240*1.5
IMG_WIDTH = 480  # 320*1.5


if __name__ == "__main__":

    output_folder = "/media/user/ssd2t/datasets2/metaworld_keyframes/all_envs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    extractor = slam_utils.get_super_glue_extractor()

    global_sample_index = 0
    
    for env_idx in range(len(env_utils.env_names)):
        if env_idx in EXCLUDED_ENV_IDS:
            continue
        
        print(env_idx, env_utils.env_names[env_idx])
        
        env, expert_policy, renderer = env_utils.generate_env_and_renderer(env_idx, height=IMG_HEIGHT, width=IMG_WIDTH)

        for sample_idx in tqdm(range(NUM_SAMPLES_PER_ENV)):
            keyframes_data = keyframe_utils.generate_keyframes(env, renderer)

            img = torch.from_numpy(keyframes_data["img"]).type(torch.uint8)
            depth = torch.from_numpy(keyframes_data["depth"]).type(torch.float32)
            seg = torch.from_numpy(keyframes_data["seg"]).type(torch.int16)
            geom_xpos = torch.from_numpy(keyframes_data["geom_xpos"]).type(torch.float32)
            geom_xmat = torch.from_numpy(keyframes_data["geom_xmat"]).type(torch.float32)

            prefix = f"sample_{env_idx}_{sample_idx}"
            torch.save(img, os.path.join(output_folder, f"{prefix}_img.tar"))
            torch.save(depth, os.path.join(output_folder, f"{prefix}_depth.tar"))
            torch.save(seg, os.path.join(output_folder, f"{prefix}_seg.tar"))
            torch.save(geom_xpos, os.path.join(output_folder, f"{prefix}_geom_xpos.tar"))
            torch.save(geom_xmat, os.path.join(output_folder, f"{prefix}_geom_xmat.tar"))


            kpts = torch.zeros((img.shape[0], 2048, 2), dtype=torch.float32)
            scores = torch.zeros((img.shape[0], 2048), dtype=torch.float32)
            descriptors = torch.zeros((img.shape[0], 2048, 256), dtype=torch.float32)

            for k in range(img.shape[0]):
                features = slam_utils.compute_features(extractor, img[k].numpy(), device=DEVICE)
                n_keypoints = features["keypoints"].shape[0]
                kpts[k][:n_keypoints] = features["keypoints"].cpu()
                scores[k][:n_keypoints] = features["keypoint_scores"].cpu()
                descriptors[k][:n_keypoints] = features["descriptors"].cpu()
            
            torch.save(kpts, os.path.join(output_folder, f"{prefix}_keypoints.tar"))
            torch.save(scores, os.path.join(output_folder, f"{prefix}_scores.tar"))
            torch.save(descriptors, os.path.join(output_folder, f"{prefix}_descriptors.tar"))
            global_sample_index += 1

    # TODO save T_pixel2world
    T_world2pixel = cu.get_camera_transform_matrix(env, "keyframes", *keyframes_data["img"][0].shape[:2])
    T_pixel2world = np.linalg.inv(T_world2pixel)
    torch.save(torch.from_numpy(T_pixel2world), os.path.join(output_folder, f"T_pixel2world.tar"))

    