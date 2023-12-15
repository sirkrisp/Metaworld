import torch.utils.data as torch_data
import torch
import numpy as np
import os

# custom imports
import utils.match_utils as match_utils


class MetaworldFeatureMatchingDataset(torch_data.Dataset):
    
    def __init__(
            self, 
            data_folder, 
            predict=False,
            env_ids: list[int] = [i for i in range(50)],
            num_samples_per_env: int = 100,
            max_keypoints: int = 100,
            eps = 0.01,
            dev=False
    ):
        super().__init__()

        self.env_ids = env_ids
        self.num_samples_per_env = num_samples_per_env
        self.max_keypoints = max_keypoints
        self.eps = eps
        self.dev = dev

        self.data_folder = data_folder
        self.predict = predict

        self.num_envs = len(self.env_ids)
        self.n_total = self.num_envs * self.num_samples_per_env
        self.T_pixel2world = self._load_torch("T_pixel2world")

        self.env_index_2_env_id = {}
        for i in range(self.num_envs):
            self.env_index_2_env_id[i] = self.env_ids[i]

    def _load_torch(self, tensor_name, prefix=None):
        if prefix:
            path = os.path.join(self.data_folder, f"{prefix}_{tensor_name}.tar")
        else:
            path = os.path.join(self.data_folder, f"{tensor_name}.tar")
        return torch.load(path)


    def __getitem__(self, index_1):
        with torch.no_grad():
            env_index = index_1 // self.num_samples_per_env
            env_id = self.env_index_2_env_id[env_index]
            sample_index_1 = index_1 % self.num_samples_per_env
            # index_1 and index_2 must be different (see explanation of filter below)
            sample_index_2 = (sample_index_1 + np.random.randint(1, self.num_samples_per_env)) % self.num_samples_per_env

            # TODO there is still an issue when k = 1
            k = 0 # np.random.randint(0, 2)
            prefix_1 = f"sample_{env_id}_{sample_index_1}"
            prefix_2 = f"sample_{env_id}_{sample_index_2}"

            # load data
            img_0 = self._load_torch("img", prefix_1)[k]
            depth_0 = self._load_torch("depth", prefix_1)[k]
            seg_0 = self._load_torch("seg", prefix_1)[k]
            geom_xpos_0 = self._load_torch("geom_xpos", prefix_1)[k]
            geom_xmat_0 = self._load_torch("geom_xmat", prefix_1)[k]
            kpts_0 = self._load_torch("keypoints", prefix_1)[k]
            dscpt_0 = self._load_torch("descriptors", prefix_1)[k]

            img_1 = self._load_torch("img", prefix_2)[k]
            depth_1 = self._load_torch("depth", prefix_2)[k]
            seg_1 = self._load_torch("seg", prefix_2)[k]
            geom_xpos_1 = self._load_torch("geom_xpos", prefix_2)[k]
            geom_xmat_1 = self._load_torch("geom_xmat", prefix_2)[k]
            kpts_1 = self._load_torch("keypoints", prefix_2)[k]
            dscpt_1 = self._load_torch("descriptors", prefix_2)[k]

            match_data = match_utils.match_features(
                self.T_pixel2world.numpy(),
                img_0.numpy(), img_1.numpy(),
                depth_0.numpy(), depth_1.numpy(),
                seg_0.numpy()[:,:,0], seg_1.numpy()[:,:,0],
                geom_xpos_0.numpy(), geom_xpos_1.numpy(),
                geom_xmat_0.numpy(), geom_xmat_1.numpy(),
                kpts_0.numpy(), kpts_1.numpy(),
                dscpt_0.numpy(), dscpt_1.numpy(),
                ngeom=geom_xpos_0.shape[0],
                max_keypoints=self.max_keypoints,
                eps=self.eps
            )

            kpts_0 = match_data["keypoints_0"]
            kpts_1 = match_data["keypoints_1"]
            dscpt_0 = match_data["descriptors_0"]
            dscpt_1 = match_data["descriptors_1"]
            matches_0 = match_data["matches_0"]
            matches_1 = match_data["matches_1"]
            assignment_mtr = match_data["assignment_mtr"]

            if self.dev:
                return img_0, img_1, depth_0, depth_1, seg_0, seg_1, match_data, kpts_0, kpts_1, dscpt_0, dscpt_1, matches_0, matches_1, assignment_mtr
            
            return kpts_0, kpts_1, dscpt_0, dscpt_1, matches_0, matches_1, assignment_mtr

    def __len__(self):
        return self.n_total    