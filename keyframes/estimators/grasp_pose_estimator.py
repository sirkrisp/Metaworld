import keyframes.seg_any as seg_any
import keyframes.mcc as mcc
import numpy as np
import torch


class GraspPoseEstimatorMCC:
    def __init__(
        self,
        seg_any_ckpt_path: str,
        mcc_ckpt_path: str,
        img_shape=(360, 480),
        device="cuda",
        log=True,
    ) -> None:
        self.log = log

        # Load seg any model
        self._log("Loading SegAny segmentor...")
        self.seg_any_estimator = seg_any.SegAny(
            ckpt_path=seg_any_ckpt_path,
            img_shape=img_shape,
        )

        # Load MCC model
        self._log("Loading MCC reconstructor...")
        self.mcc_reconstructor = mcc.ReconstructMCC(
            mcc_ckpt_path=mcc_ckpt_path,
            device=device,
        )

        self.step_data = {}

    def _log(self, msg):
        if self.log:
            print(msg)

    def predict_grasp_point_stepwise(
        self, step, img, xyz, kpts, mcc_params={"grid_granularity": 0.025}
    ):
        """
        Args:
            step (int): 0 = compute mask, 1 = compute MCC, 2 = generate potential grasp points, 3 = compute grasp point
            img: (3, H, W) np.ndarray
            xyz: (H*W, 3) np.ndarray
            kpts: (N, 2) np.ndarray
        Returns:
            step_data (dict): contains intermediate results
        """
        if step == 0:
            self.step_data = {}
            self.step_data["seg"] = self.seg_any_estimator.compute_mask(img, kpts)
        elif step == 1:
            self.mcc_reconstructor.set_model_input(
                img,
                self.step_data["seg"],
                xyz,
                grid_granularity=mcc_params["grid_granularity"],
            )
            (
                self.step_data["pred_occupy"],
                self.step_data["pred_colors"],
                self.step_data["occupy_threshold"],
                self.step_data["verts"],
                self.step_data["faces"],
                self.step_data["vert_normals"],
                self.step_data["values"],
            ) = self.mcc_reconstructor.reconstruct()
        elif step == 2:
            (
                self.step_data["verts_1_new"],
                self.step_data["verts_2_new"],
                self.step_data["verts_1"],
                self.step_data["verts_2"],
                self.step_data["v1_to_v2"],
            ) = self._generate_potential_grasp_points(
                self.step_data["verts"],
                self.step_data["vert_normals"],
                occupy_threshold=self.step_data["occupy_threshold"],
                scale=1 / self.mcc_reconstructor.masked_max_dist,
            )
        elif step == 3:
            self.step_data["grasp_point"] = self._compute_grasp_point(
                self.step_data["verts_1_new"], self.step_data["verts_2_new"]
            )
        return self.step_data

    def predict_grasp_point(
        self, img, xyz, kpts, mcc_params={"grid_granularity": 0.025}
    ):
        """
        Args:
            img: (3, H, W) np.ndarray
            xyz: (H*W, 3) np.ndarray
            kpts: (N, 2) np.ndarray
        Returns:
            grasp_point: (3,) np.ndarray
        """
        self._log("Computing mask...")
        seg = self.seg_any_estimator.compute_mask(img, kpts)
        self._log("Computing MCC...")
        self.mcc_reconstructor.set_model_input(
            img, seg, xyz, grid_granularity=mcc_params["grid_granularity"]
        )
        (
            pred_occupy,
            pred_colors,
            occupy_threshold,
            verts,
            faces,
            vert_normals,
            values,
        ) = self.mcc_reconstructor.reconstruct()
        self._log("Generating potential grasp points...")
        verts_1_new, verts_2_new, _, _, _ = self._generate_potential_grasp_points(
            verts,
            vert_normals,
            occupy_threshold=occupy_threshold,
            scale=1 / self.mcc_reconstructor.masked_max_dist,
        )
        # self._log("Collision check for potential grasp points...")
        # has_collision, pred_occupy, sample_points = self._check_collision(verts_1_new, verts_2_new, occupy_threshold)
        self._log("Selecting grasp points closes to center of moving object...")
        if verts_1_new.shape[0] == 0 or verts_2_new.shape[0] == 0:
            grasp_point = None
            self._log("No grasp point found.")
        else:
            grasp_point = self._compute_grasp_point(verts_1_new, verts_2_new)
        return {
            "grasp_point": grasp_point,
            "seg": seg,
            "pred_occupy": pred_occupy,
            "pred_colors": pred_colors,
            "occupy_threshold": occupy_threshold,
            "verts": verts,
            "faces": faces,
            "vert_normals": vert_normals,
            "values": values,
        }

    def _compute_grasp_point(
        self, contact_verts_1: np.ndarray, contact_verts_2: np.ndarray
    ):
        """
        Args:
            contact_verts_1: np.ndarray (n_contact_points, 3)
            contact_verts_2: np.ndarray (n_contact_points, 3)
            has_collision: np.ndarray (n_contact_points)
        """
        moving_obj_center = self.mcc_reconstructor.masked_mean
        # grasp_points = contact_verts_1[~has_collision] + (contact_verts_2[~has_collision] - contact_verts_1[~has_collision]) / 2
        grasp_points = contact_verts_1 + (contact_verts_2 - contact_verts_1) / 2
        grasp_points = self.mcc_reconstructor.denormalize(grasp_points)
        grasp_point = grasp_points[
            np.argmin(np.linalg.norm(grasp_points - moving_obj_center, axis=1))
        ]
        return grasp_point

    def _check_collision_v2(
        self,
        contact_verts: np.ndarray,
        occupy_threshold: float,
        site: str,  # "left" or "right"
        gripper_offset_y: float = 0.001,
    ):
        n_points_per_dim = 10
        n_sample_points = (
            n_points_per_dim * n_points_per_dim
        )  # gripper_points_sampler.get_num_sample_points()

        n_contact_points = contact_verts.shape[0]
        sample_points = np.zeros((n_sample_points * n_contact_points, 3))

        contact_verts_unnormalized = self.mcc_reconstructor.denormalize(contact_verts)

        # sample gripper points for each contact point
        gripper_pad_width = 0.03
        # gripper_depth = 0.006
        # TODO make more robust => error handling or make input param
        gripper_offset = np.array([0, gripper_offset_y, 0])
        if site == "left":
            gripper_offset = -gripper_offset
        for i in range(n_contact_points):
            sample_points[
                i * n_sample_points : (i + 1) * n_sample_points
            ] = sample_xz_plane_grid(
                contact_verts_unnormalized[i] + gripper_offset,
                gripper_pad_width,
                n_points_per_dim,
            )

        sample_points = self.mcc_reconstructor.normalize(sample_points)

        # query model
        pred_occupy, _ = self.mcc_reconstructor.query_model(
            torch.from_numpy(sample_points)
        )
        pred_occupy = pred_occupy.cpu().numpy()

        has_collision = np.zeros(n_contact_points, dtype=bool)

        for i in range(n_contact_points):
            has_collision[i] = np.any(
                pred_occupy[i * n_sample_points : (i + 1) * n_sample_points]
                > occupy_threshold
            )
        return has_collision, pred_occupy, sample_points

    def _check_collision(
        self,
        # left and right gripper pad contact points
        contact_verts_1: np.ndarray,
        contact_verts_2: np.ndarray,
        occupy_threshold=5,
    ):
        """Check if gripper collides with object if gripper is placed at contact_verts_1/contact_verts_2
        Args:
            contact_verts_1: np.ndarray (n_contact_points, 3)
            contact_verts_2: np.ndarray (n_contact_points, 3)
            occupy_threshold: float
        Returns:
            has_collision: np.ndarray (n_contact_points)
            pred_occupy: np.ndarray (n_contact_points * n_sample_points)
            sample_points: np.ndarray (n_contact_points * n_sample_points, 3)
        """
        assert contact_verts_1.shape == contact_verts_2.shape

        # gripper sampler
        # gripper_points_sampler = gripper_sample_points.GripperSampler(cell_size=0.025)

        n_points_per_dim = 10
        n_sample_points = (
            2 * n_points_per_dim * n_points_per_dim
        )  # gripper_points_sampler.get_num_sample_points()

        n_contact_points = contact_verts_1.shape[0]
        sample_points = np.zeros((n_sample_points * n_contact_points, 3))

        contact_verts_1_unnormalized = self.mcc_reconstructor.denormalize(
            contact_verts_1
        )
        contact_verts_2_unnormalized = self.mcc_reconstructor.denormalize(
            contact_verts_2
        )

        # sample gripper points for each contact point
        for i in range(n_contact_points):
            verts_pair = np.vstack(
                [contact_verts_1_unnormalized[i], contact_verts_2_unnormalized[i]]
            )
            # offset = verts_pair[0] + (verts_pair[1] - verts_pair[0]) / 2 # + np.array([0,0,-0.015])
            # pad_dist = verts_pair[1,1] - verts_pair[0,1] + 0.000
            # sample_points[i*n_sample_points:(i+1)*n_sample_points] = gripper_points_sampler.sample(pad_dist, offset)[:, :3]
            gripper_offset = np.array([0, 0.0005, 0])
            sample_left = sample_xz_plane_grid(
                verts_pair[0] - gripper_offset, 0.01, n_points_per_dim
            )
            sample_right = sample_xz_plane_grid(
                verts_pair[1] + gripper_offset, 0.01, n_points_per_dim
            )
            sample_points[
                i * n_sample_points : (i + 1) * n_sample_points
            ] = np.concatenate([sample_left, sample_right], axis=0)

        sample_points = self.mcc_reconstructor.normalize(sample_points)

        # query model
        pred_occupy, _ = self.mcc_reconstructor.query_model(
            torch.from_numpy(sample_points)
        )
        pred_occupy = pred_occupy.cpu().numpy()

        has_collision = np.zeros(n_contact_points, dtype=bool)
        for i in range(n_contact_points):
            has_collision[i] = np.any(
                pred_occupy[i * n_sample_points : (i + 1) * n_sample_points]
                > occupy_threshold
            )
        return has_collision, pred_occupy, sample_points

    def _generate_potential_grasp_points(
        self,
        verts: np.ndarray,
        vert_normals: np.ndarray,
        occupy_threshold: float,
        scale: float,
    ):
        """
        Args:
            verts: (N, 3)
            vert_normals: (N, 3)
        Returns:
            verts_1_new: (N, 3)
            verts_2_new: (N, 3)
            verts_1: (N, 3)
            verts_2: (N, 3)
            v1_to_v2: (N,) np.ndarray
        """
        target_normal = np.array([0, 1, 0])
        gripper_max_width = 0.088 * scale

        # 1) only consider vertices that are aligned with target_normal
        angle_err = 20
        while True:
            angle_1 = calculate_angle(vert_normals, target_normal)
            angle_2 = calculate_angle(vert_normals, -target_normal)
            mask_1 = angle_1 < angle_err
            mask_2 = angle_2 < angle_err
            if np.sum(mask_1) > 0 and np.sum(mask_2) > 0:
                break
            angle_err += 1
        verts_1 = verts[mask_1]
        verts_2 = verts[mask_2]

        # TODO Use curvature at vertices as well (curvature should be pos/neg depending on site)
        # => somewhat considered in Mesh - Point Cloud collision checking

        # 2) collision checking
        gripper_offset_y = 0.001
        max_tries = 100
        for _ in range(max_tries):
            has_collision_1, _, _ = self._check_collision_v2(
                verts_1,
                occupy_threshold=occupy_threshold,
                site="left",
                gripper_offset_y=gripper_offset_y,
            )
            has_collision_2, _, _ = self._check_collision_v2(
                verts_2,
                occupy_threshold=occupy_threshold,
                site="right",
                gripper_offset_y=gripper_offset_y,
            )

            if np.sum(~has_collision_1) == 0 or np.sum(~has_collision_2) == 0:
                gripper_offset_y += 0.001
                continue

            verts_1 = verts_1[~has_collision_1, :]
            verts_2 = verts_2[~has_collision_2, :]

            # 2) find vertices on the opposite side
            v1_to_v2 = -np.ones(verts_1.shape[0], dtype=np.int64)
            v2_indices = np.arange(verts_2.shape[0])
            for i in range(verts_1.shape[0]):
                dist_xyz = verts_2 - verts_1[i]
                dist_xyz_abs = np.abs(dist_xyz)
                # NOTE these masks could still mask out all vertices even if no vertices collide
                mask_x = dist_xyz_abs[:, 0] < (0.005 * scale)
                mask_y = np.logical_and(
                    dist_xyz_abs[:, 1] < gripper_max_width, dist_xyz[:, 1] > 0
                )
                # due to reconstruction error, we need to allow for some error in z direction
                # TODO ==> another assumption => check implications and scenarios where this might fail
                mask_z = dist_xyz_abs[:, 2] < (0.1 * scale)
                mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
                if np.sum(mask) > 0:
                    v1_to_v2[i] = v2_indices[mask][0]

            if np.sum(v1_to_v2 != -1) > 0:
                break

            gripper_offset_y += 0.001
        verts_1_new = verts_1[v1_to_v2 != -1]
        verts_2_new = verts_2[v1_to_v2[v1_to_v2 != -1]]
        return verts_1_new, verts_2_new, verts_1, verts_2, v1_to_v2


# ================
# Utils
# ================


def sample_xz_plane_grid(center_pt, grid_size, n_points_per_dim):
    """Sample grid of points in xz plane
    Args:
        center_pt: np.ndarray (3)
        grid_size: float
        n_points_per_dim: int
    """
    x = np.linspace(
        center_pt[0] - grid_size / 2, center_pt[0] + grid_size / 2, n_points_per_dim
    )
    z = np.linspace(
        center_pt[2] - grid_size / 2, center_pt[2] + grid_size / 2, n_points_per_dim
    )
    x, z = np.meshgrid(x, z)
    y = np.ones_like(x) * center_pt[1]
    grid = np.stack([x, y, z], axis=-1)
    return grid.reshape(-1, 3)


def calculate_angle(a, b):
    # Normalize vectors
    a_normalized = np.linalg.norm(a, axis=1)
    b_normalized = np.linalg.norm(b)

    # Calculate dot product
    dot_product = np.dot(a, b)

    # Calculate cosine similarity
    cosine_similarity = dot_product / (a_normalized * b_normalized)

    # Calculate angle in radians
    angle_radians = np.arccos(cosine_similarity)

    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees
