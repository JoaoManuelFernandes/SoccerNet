import cv2
import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from yacs.config import CfgNode as CN

from tracklab.pipeline import DetectionLevelModule
# FIXME this should be removed and use KeypointsSeriesAccessor and KeypointsFrameAccessor
from tracklab.utils.coordinates import rescale_keypoints
from tracklab.utils.collate import default_collate
from sn_gamestate.reid.prtreid_dataset import ReidDataset
from prtreid.scripts.main import build_config, build_torchreid_model_engine
from prtreid.tools.feature_extractor import FeatureExtractor
from prtreid.utils.imagetools import (
    build_gaussian_heatmaps,
)
from tracklab.utils.collate import Unbatchable

import tracklab
from pathlib import Path


import prtreid
from torch.nn import functional as F
from prtreid.data.masks_transforms import (
    CocoToSixBodyMasks,
    masks_preprocess_transforms,
)
from prtreid.utils.tools import extract_test_embeddings
from prtreid.data.datasets import configure_dataset_class

from prtreid.scripts.default_config import engine_run_kwargs

from tracklab.utils.download import download_file


_TORSO_PART_IDX = 1   # CocoToSixBodyMasks: head=0, torso=1, left_arm=2, right_arm=3, legs=4,5
_MASK_THRESHOLD = 0.25  # min BPBreID attention score to include a pixel


def _extract_jersey_color_features(crop_rgb: np.ndarray,
                                   body_mask_torso: np.ndarray | None = None) -> dict:
    """
    Extract jersey color features using BPBreID torso mask (Option B) when available,
    falling back to fixed 25-75% spatial crop (Option A).

    Option B (body_mask_torso provided):
      Resizes the BPBreID torso attention map to the crop dimensions and uses it
      as a pixel mask — isolates the jersey fabric, excluding head, arms, legs, skin.
    Option A (fallback):
      Uses rows 25-75% of crop height — removes grass/shoes but not arms/skin.

    Returns 27-dim float32 vector: 24 (HSV hist) + 3 (Lab mean, normalised 0-1).
    """
    zeros = {
        "color_vec": np.zeros(27, dtype=np.float32),
        "mean_rgb": np.zeros(3, dtype=np.float32),
        "mean_hsv": np.zeros(3, dtype=np.float32),
        "mean_lab": np.zeros(3, dtype=np.float32),
    }
    h, w = crop_rgb.shape[:2]
    if h < 8 or w < 4:
        return zeros

    crop_bgr = np.ascontiguousarray(crop_rgb[:, :, ::-1])
    crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    crop_lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab)

    # Build pixel selection mask
    pixel_mask = None
    if body_mask_torso is not None and body_mask_torso.max() > 0:
        resized = cv2.resize(body_mask_torso.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_LINEAR)
        resized /= resized.max()
        pixel_mask = (resized >= _MASK_THRESHOLD).astype(np.uint8)

    if pixel_mask is None or pixel_mask.sum() < 16:  # fallback: not enough masked pixels
        pixel_mask = np.zeros((h, w), dtype=np.uint8)
        y0, y1 = max(0, int(0.25 * h)), min(h, int(0.75 * h))
        pixel_mask[y0:y1, :] = 1

    # Per-channel means on selected pixels
    masked_rgb = crop_rgb[pixel_mask > 0]
    masked_hsv = crop_hsv[pixel_mask > 0]
    masked_lab = crop_lab[pixel_mask > 0]

    mean_rgb = masked_rgb.mean(0).astype(np.float32) / 255.0          if len(masked_rgb) else np.zeros(3, np.float32)
    mean_hsv = masked_hsv.mean(0).astype(np.float32) / np.array([180., 255., 255.]) if len(masked_hsv) else np.zeros(3, np.float32)
    mean_lab = masked_lab.mean(0).astype(np.float32) / 255.0          if len(masked_lab) else np.zeros(3, np.float32)

    # HSV histogram on selected pixels only
    hist_parts = []
    for ch, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
        hist = cv2.calcHist([crop_hsv], [ch], pixel_mask, [8], [lo, hi]).flatten().astype(np.float32)
        total = hist.sum()
        hist_parts.append(hist / total if total > 0 else hist)
    hsv_hist = np.concatenate(hist_parts)  # 24 dims

    color_vec = np.concatenate([hsv_hist, mean_lab])  # 27 dims
    return {"color_vec": color_vec, "mean_rgb": mean_rgb, "mean_hsv": mean_hsv, "mean_lab": mean_lab}


class PRTReId(DetectionLevelModule):
    collate_fn = default_collate
    input_columns = ["bbox_ltwh"]
    output_columns = [
        "embeddings", "embeddings_with_colors",
        "visibility_scores", "body_masks",
        "role_detection", "role_confidence",
        "jersey_mean_rgb", "jersey_mean_hsv", "jersey_mean_lab",
    ]
    forget_columns = ["embeddings", "body_masks"]
    role_mapping = {'ball': 0, 'goalkeeper': 1, 'other': 2, 'player': 3, 'referee': 4, None: -1}

    def __init__(
        self,
        cfg,
        tracking_dataset,
        dataset,
        device,
        save_path,
        job_id,
        use_keypoints_visibility_scores_for_reid,
        training_enabled,
        batch_size,
    ):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        tracking_dataset.name = dataset.name
        tracking_dataset.nickname = dataset.nickname
        self.dataset_cfg = dataset
        self.use_keypoints_visibility_scores_for_reid = (
            use_keypoints_visibility_scores_for_reid
        )
        tracking_dataset.name = self.dataset_cfg.name
        tracking_dataset.nickname = self.dataset_cfg.nickname
        additional_args = {
            "tracking_dataset": tracking_dataset,
            "reid_config": self.dataset_cfg,
            "role_mapping": self.role_mapping,
            "pose_model": None,
        }
        prtreid.data.register_image_dataset(
            tracking_dataset.name,
            configure_dataset_class(ReidDataset, **additional_args),
            tracking_dataset.nickname,
        )
        self.cfg = CN(OmegaConf.to_container(cfg, resolve=True))
        self.download_models(load_weights=self.cfg.model.load_weights,
                             pretrained_path=self.cfg.model.bpbreid.hrnet_pretrained_path,
                             backbone=self.cfg.model.bpbreid.backbone)
        self.inverse_role_mapping = {v: k for k, v in self.role_mapping.items()}
        # set parts information (number of parts K and each part name),
        # depending on the original loaded masks size or the transformation applied:
        self.cfg.data.save_dir = save_path
        self.cfg.project.job_id = job_id
        self.cfg.use_gpu = torch.cuda.is_available()
        self.cfg = build_config(config=self.cfg)
        self.test_embeddings = self.cfg.model.bpbreid.test_embeddings
        # Register the PoseTrack21ReID dataset to Torchreid that will be instantiated when building Torchreid engine.
        self.training_enabled = training_enabled
        self.feature_extractor = None
        self.model = None

    def download_models(self, load_weights, pretrained_path, backbone):
        if Path(load_weights).name == "prtreid-soccernet-baseline.pth.tar":
            md5 = "9633825232bc89f23a94522c5561650e"
            download_file("https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1",
                          local_filename=load_weights, md5=md5)
        if backbone == "hrnet32":
            md5 = "58ea12b0420aa3adaa2f74114c9f9721"
            path = Path(pretrained_path) / "hrnetv2_w32_imagenet_pretrained.pth"
            download_file("https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1",
                          local_filename=path, md5=md5)
    @torch.no_grad()
    def preprocess(
        self, image, detection: pd.Series, metadata: pd.Series
    ):  # Tensor RGB (1, 3, H, W)
        mask_w, mask_h = 32, 64
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }

        return batch


    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        im_crops = batch["img"]
        im_crops = [im_crop.cpu().detach().numpy() for im_crop in im_crops]

        if "masks" in batch:
            external_parts_masks = batch["masks"].cpu().detach().numpy()
        else:
            external_parts_masks = None

        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.cfg,
                model_path=self.cfg.model.load_weights,
                device=self.device,
                image_size=(self.cfg.data.height, self.cfg.data.width),
                model=self.model,
                verbose=False,
            )

        reid_result = self.feature_extractor(im_crops, external_parts_masks=external_parts_masks)

        embeddings, visibility_scores, body_masks, _, role_cls_scores = extract_test_embeddings(
            reid_result, self.test_embeddings
        )

        role_scores_ = []
        role_scores_.append(role_cls_scores['globl'].cpu() if role_cls_scores is not None else None)
        role_scores_ = torch.cat(role_scores_, 0) if role_scores_[0] is not None else None
        roles = [torch.argmax(i).item() for i in role_scores_] if role_scores_ is not None else []
        roles = [self.inverse_role_mapping[index] for index in roles]
        role_confidence = [torch.max(i).item() for i in role_scores_] if role_scores_ is not None else []

        embeddings = embeddings.cpu().detach().numpy()
        visibility_scores = visibility_scores.cpu().detach().numpy()
        body_masks = body_masks.cpu().detach().numpy()

        if self.use_keypoints_visibility_scores_for_reid:
            kp_visibility_scores = batch["visibility_scores"].numpy()
            if visibility_scores.shape[1] > kp_visibility_scores.shape[1]:
                kp_visibility_scores = np.concatenate(
                    [np.ones((visibility_scores.shape[0], 1)), kp_visibility_scores],
                    axis=1,
                )
            visibility_scores = np.float32(kp_visibility_scores)

        # Extract jersey color using BPBreID torso mask when available (Option B),
        # falling back to fixed 25-75% spatial crop (Option A).
        color_results = []
        for i, crop in enumerate(im_crops):
            torso_mask = None
            if (body_masks is not None
                    and i < body_masks.shape[0]
                    and body_masks.shape[1] > _TORSO_PART_IDX):
                torso_mask = body_masks[i, _TORSO_PART_IDX]  # (H_feat, W_feat)
            color_results.append(_extract_jersey_color_features(crop, torso_mask))
        color_vecs  = np.stack([r["color_vec"]  for r in color_results], axis=0)  # (N, 27)
        jersey_rgb  = np.stack([r["mean_rgb"]   for r in color_results], axis=0)  # (N, 3)
        jersey_hsv  = np.stack([r["mean_hsv"]   for r in color_results], axis=0)  # (N, 3)
        jersey_lab  = np.stack([r["mean_lab"]   for r in color_results], axis=0)  # (N, 3)

        # Flatten embeddings to 2D (may be 3D when multi-part: batch × parts × dim)
        emb_2d = embeddings.reshape(embeddings.shape[0], -1)

        # embeddings_with_colors = ReID (256) + color (27) = 283 dims
        embeddings_with_colors = np.concatenate([emb_2d, color_vecs], axis=1)

        reid_df = pd.DataFrame(
            {
                "embeddings":             list(embeddings),
                "embeddings_with_colors": list(embeddings_with_colors),
                "visibility_scores":      list(visibility_scores),
                "body_masks":             list(body_masks),
                "role_detection":         roles,
                "role_confidence":        role_confidence,
                "jersey_mean_rgb":        list(jersey_rgb),
                "jersey_mean_hsv":        list(jersey_hsv),
                "jersey_mean_lab":        list(jersey_lab),
            },
            index=detections.index,
        )
        return reid_df


    def train(self):
        self.engine, self.model = build_torchreid_model_engine(self.cfg)
        self.engine.run(**engine_run_kwargs(self.cfg))
