import utils.inference_utils as inference_utils
import keyframes.pl_modules as pl_modules
import utils.slam_utils as slam_utils
import utils.depth_utils as depth_utils
import utils.predict_utils as predict_utils
import keyframes.grasp_pose as grasp_pose
import keyframes.policies.pick_place_policy as pick_place_policy
import numpy as np


class KeyframePolicy:
    """
    NOTE
    Example usage:
    ```
    # initialize
    keyframe_policy = KeyframePolicy(
        fts_match_ckpt_path="...",
        seg_any_ckpt_path="...",
        mcc_ckpt_path="...",
        T_world2pixel=np.array(...),
        device="cuda",
    )

    # set keyframes
    img_ref_0 = ...
    img_ref_1 = ...
    depth_ref_0 = ...
    depth_ref_1 = ...
    keyframe_policy.set_keyframes(img_ref_0, img_ref_1, depth_ref_0, depth_ref_1)

    # set current view
    img_cur = ...
    depth_cur = ...
    keyframe_policy.set_current_view(img_cur, depth_cur)

    # get action
    obs = ...
    action = keyframe_policy.get_action(obs)
    """

    def __init__(self, fts_match_ckpt_path: str, seg_any_ckpt_path: str, mcc_ckpt_path: str, T_world2pixel: np.ndarray, device="cuda", log=True) -> None:

        self.log = log
        self.device = device
        self.T_world2pixel = T_world2pixel

        # load feature extractor
        self.model_fts_extract = slam_utils.get_super_glue_extractor(device=device)

        # load feature matching model
        config = inference_utils.load_config("feature_matching", "01")
        self.model_fts_match = pl_modules.PLModel(config["model_args"], config["optimizer_args"])
        self.model_fts_match = self.model_fts_match.load_from_checkpoint(fts_match_ckpt_path)
        self.model_fts_match.to(device)
        self.model_fts_match.eval()

        self.match_data_0 = None
        self.pred_res = None

        # grasp pose
        self.grasp_pose_estimator = grasp_pose.GraspPoseEstimatorMCC(
            seg_any_ckpt_path=seg_any_ckpt_path,
            mcc_ckpt_path=mcc_ckpt_path,
            device=device,
            log=log,
        )

        self.policy = None


    def set_keyframes(self, img_ref_0, img_ref_1, depth_ref_0, depth_ref_1):
        self._reset()
        self.img_ref_0 = img_ref_0
        self.img_ref_1 = img_ref_1
        self.depth_ref_0 = depth_ref_0
        self.depth_ref_1 = depth_ref_1
        fts_ref_0 = slam_utils.compute_features(self.model_fts_extract, img_ref_0)
        fts_ref_1 = slam_utils.compute_features(self.model_fts_extract, img_ref_1)
        self.kpts_ref_0 = fts_ref_0["keypoints"]
        self.kpts_ref_1 = fts_ref_1["keypoints"]
        self.desc_ref_0 = fts_ref_0["descriptors"]
        self.desc_ref_1 = fts_ref_1["descriptors"]
        self.kpts_scores_ref_0 = fts_ref_0["keypoint_scores"]
        self.kpts_scores_ref_1 = fts_ref_1["keypoint_scores"]
        # 0) match between ref_0 and ref_1
        # TODO get_model_input_v2: take fts as input
        model_input_0 = inference_utils.get_model_input_v2(self.kpts_ref_0, self.kpts_ref_1, self.desc_ref_0, self.desc_ref_1)
        self.match_data_0 = inference_utils.match_lightglue_finetuned_v2(self.model_fts_match, model_input_0)


    def set_current_view(self, img_cur, depth_cur):
        assert self.match_data_0 is not None, "set_keyframes() must be called before set_current_view()"
        self.img_cur = img_cur
        self.depth_cur = depth_cur

        # compute features
        fts_cur = slam_utils.compute_features(self.model_fts_extract, img_cur)
        self.kpts_cur = fts_cur["keypoints"]
        self.desc_cur = fts_cur["descriptors"]
        self.kpts_scores_cur = fts_cur["keypoint_scores"]

        # predict feature movement
        self.pred_res = inference_utils.predict_feature_movement_v2(
            self.model_fts_match,
            depth_ref_0=self.depth_ref_0,
            depth_ref_1=self.depth_ref_1,
            depth_cur=self.depth_cur,
            desc_ref_0=self.desc_ref_0,
            desc_cur=self.desc_cur,
            kpts_ref_0=self.kpts_ref_0,
            kpts_ref_1=self.kpts_ref_1,
            kpts_cur=self.kpts_cur,
            T_world2pixel=self.T_world2pixel,
            match_data_0=self.match_data_0,
        )

        # create policy based on feature movement
        self.policy = self._get_policy(img_cur, depth_cur, self.pred_res)


    def _reset(self):
        self.pred_res = None
        self.policy = None


    def _log(self, msg):
        if self.log:
            print(msg)


    def _get_policy(self, img_cur, depth_cur, pred_res):
        """ predict policy based on pred_res
        """
        policy_type = self._get_policy_type()
        if policy_type == "pick_place":
            return self._get_pick_place_policy(img_cur, depth_cur, pred_res)
        raise NotImplementedError

    
    def _get_policy_type(self):
        """ predict policy type based on pred_res
        """
        return "pick_place"

    
    def _get_pick_place_policy(self, img_cur, depth_cur, pred_res):
        xyz = depth_utils.pixel_coords_to_world_coords(
            self.T_pixel2world, 
            depth_cur,
        )[:,:3]
        grasp_point_res = self.grasp_pose_estimator.predict_grasp_point(img_cur, xyz, pred_res["kpts"]["kpts_2_0"])
        R_closest, t_closest, t_no_rot_closest, x_center, moving_fts_inter_frame_transform = predict_utils.compute_T(
            pred_res["kpts"]["wpos_kpts_2_0"], 
            pred_res["kpts"]["wpos_kpts_2_1"]
        )
        grasp_point = grasp_point_res["grasp_point"]
        target_point = moving_fts_inter_frame_transform(grasp_point)
        return pick_place_policy.PickPlacePolicy(grasp_point, target_point)


    def get_action(self, obs):
        assert self.policy_type is not None, "set_current_view() must be called before get_action()"
        return self.policy.get_action(obs)