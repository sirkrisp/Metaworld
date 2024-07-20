from typing import Dict, List, Union
import torch
import torch.utils.data as torch_data
import numpy as np

from keyframes.align_ft_spair.utils import geo_utils, spair_utils


class SPairEmbdDataset(torch_data.Dataset):
    def __init__(
        self,
        spair_data_folder: str,
        embds_folder_dino: str,
        embds_folder_sd: str,
        img_files_np_path: str,
        kpt_indices: List[int],
        category="aeroplane",
        mode="train",
        img_size=960,
        embd_size=60,
        pad=True,
    ):
        # load image files
        self.img_files = np.load(img_files_np_path).tolist()
        self.n_imgs = len(self.img_files)
        self.data_folder = spair_data_folder
        self.flips = [False, True] if mode == "train" else [False]
        self.kpt_indices = kpt_indices

        # load embeddings
        img_embds, img_embds_hat = geo_utils.load_geo_embds(
            self.img_files,
            embds_folder_dino=embds_folder_dino,
            embds_folder_sd=embds_folder_sd,
            flips=self.flips,
        )
        img_embds = img_embds.detach().cpu()
        img_embds_hat = img_embds_hat.detach().cpu()

        # extract embeddings at embedding coords of keypoints
        kpt_embd_coords: List[torch.Tensor] = []
        for flip in self.flips:
            kpt_embd_coords += spair_utils.load_kpt_embd_coords(
                img_files=self.img_files,
                spair_data_folder=spair_data_folder,
                img_new_size=img_size,
                embd_size=embd_size,
                angle_deg=0,
                flip=1 if flip else 0,
                pad=pad,
                # TODO needs to be adjusted for other cetgories!!!
                get_flipped_kpt_index=spair_utils.aeroplane_get_flipped_kpt_index,
            )

        # some basic checks
        assert img_embds.shape[0] == len(kpt_embd_coords)
        if len(self.flips) == 2:
            for i in range(self.n_imgs):
                if (
                    kpt_embd_coords[i].shape[0]
                    != kpt_embd_coords[i + self.n_imgs].shape[0]
                ):
                    print(i, kpt_embd_coords[i].shape)

        # only consider img_files where all kpt_indices are present
        sample_selection = []
        for i, kpt_embd_coords_i in enumerate(kpt_embd_coords):
            if kpt_embd_coords_i.size(0) > 0:
                kpt_indices_i = kpt_embd_coords_i[:,2].tolist()
                if all(kpt_index in kpt_indices_i for kpt_index in kpt_indices):
                    sample_selection.append(i)
        self.n_imgs = len(sample_selection)
        img_embds = img_embds[sample_selection]
        img_embds_hat = img_embds_hat[sample_selection]
        kpt_embd_coords = [kpt_embd_coords[i] for i in sample_selection]

        # for each image collect features at embeddings coords of keypoints
        self.img_kpt_embds: List[torch.Tensor] = []
        for i, embd_coords in enumerate(kpt_embd_coords):
            if embd_coords.size(0) > 0:
                kpt_embds = []
                for kpt_index in kpt_indices:
                    kpt_embds.append(img_embds_hat[
                        # i, :, embd_coords[embd_coords[:,2] == kpt_index][kpt_indices, 1], embd_coords[kpt_indices, 0]
                        i, :, embd_coords[embd_coords[:,2] == kpt_index][0, 1], embd_coords[0, 0]
                    ])
                kpt_embds = torch.stack(kpt_embds, dim=0).T # shape (C,len(kpt_indices))
                self.img_kpt_embds.append(kpt_embds)

        # # collect features at embeddings coords of keypoints
        # max_n_kpts = 30
        # kpt_idx_to_kpt_embds_list: Dict[int,List[torch.Tensor]] = {i: [] for i in range(max_n_kpts)}
        # for i, embd_coords in enumerate(kpt_embd_coords):
        #     if embd_coords.size(0) > 0:
        #         kpt_embds = img_embds_hat[
        #             i, :, embd_coords[:, 1], embd_coords[:, 0]
        #         ]  # shape (C, n_kpts)
        #         for j, kpt_idx in enumerate(embd_coords[:, 2].tolist()):
        #             kpt_idx_to_kpt_embds_list[kpt_idx].append(kpt_embds[:, j])
        # # kpt_idx_to_kpt_embds[kpt_index] = stacked features corresponding to keypoints with kpt_index
        # kpt_idx_to_kpt_embds: Dict[int, Union[torch.Tensor, None]] = {}
        # for i in range(max_n_kpts):
        #     if len(kpt_idx_to_kpt_embds_list[i]) > 0:
        #         kpt_idx_to_kpt_embds[i] = torch.stack(kpt_idx_to_kpt_embds_list[i])
        #     else:
        #         kpt_idx_to_kpt_embds[i] = None

        # # stack all kpt embeddings and create tensor with categories
        # all_kpt_embds = []
        # all_kpt_indices = []
        # self.n_total_kpts = 0
        # for kpt_index in self.kpt_indices:
        #     kpt_embds = kpt_idx_to_kpt_embds[kpt_index]
        #     if kpt_embds is not None:
        #         all_kpt_embds.append(kpt_embds)
        #         print("kpt_embds.shape of", kpt_index, kpt_embds.shape)
        #         all_kpt_indices += [kpt_index] * kpt_embds.size(0)
        #         self.n_total_kpts += kpt_embds.size(0)
        # self.all_kpt_embds = all_kpt_embds
        # # self.all_kpt_embds: torch.Tensor = torch.cat(all_kpt_embds, dim=0)
        # self.all_kpt_indices: torch.Tensor = torch.tensor(all_kpt_indices)

    def __len__(self):
        return self.n_imgs**2

    def __getitem__(self, _: int):
        index1 = np.random.choice(self.n_imgs)
        kpt_embds_1 = self.img_kpt_embds[index1]
        index2 = (index1 + np.random.choice(self.n_imgs-1)) % self.n_imgs
        kpt_embds_2 = self.img_kpt_embds[index2]
        return kpt_embds_1, kpt_embds_2
        
    # def __len__(self):
    #     return self.n_total_kpts**2

    # def __getitem__(self, _: int):
    #     kpt1_index, kpt2_index = np.random.choice(len(self.kpt_indices), 2, replace=False)
    #     kpt1_i1, kpt1_i2 = np.random.choice(self.all_kpt_embds[kpt1_index].size(0), 2, replace=False)
    #     kpt2_i = np.random.choice(self.all_kpt_embds[kpt2_index].size(0))
    #     embd1_1 = self.all_kpt_embds[kpt1_index][kpt1_i1]
    #     embd1_2 = self.all_kpt_embds[kpt1_index][kpt1_i2]
    #     embd2 = self.all_kpt_embds[kpt2_index][kpt2_i]
    #     return embd1_1, embd1_2, embd2
