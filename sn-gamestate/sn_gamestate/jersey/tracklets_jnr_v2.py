from __future__ import annotations

"""
tracklets_jnr_v2.py
====================
Pipeline JNR (Jersey Number Recognition) v2 com score de legibilidade composto.

Arquitectura:
  1. YOLO pose → gate de costas (S_back contínuo)
  2. Score de legibilidade S_leg = w1*S_back + w2*S_size + w3*S_blur
                                  + w4*S_contrast + w5*S_occ + w6*S_yaw
  3. Selecção top-K frames por S_leg dentro de cada tracklet
  4. ROI das costas (entre ombros, descendo pelo torso)
  5. PaddleOCR rec-only nos crops bons
  6. Fusão temporal ponderada por S_leg * ocr_conf
  7. Fallback Qwen2.5-VL (stub; activar com --qwen-fallback quando disponível)

Thresholds de legibilidade:
  S_leg < 0.45        → ignorar frame
  0.45 <= S_leg < 0.65 → baixa prioridade (guardado, mas não vai a OCR por omissão)
  0.65 <= S_leg < 0.80 → OCR normal
  S_leg >= 0.80       → frame premium

Run:
  conda activate tracklab
  python tracklets_jnr_v2.py \
    --tracklets-dir /path/to/tracklets_images \
    --out-dir /path/to/out \
    --pose-weights yolo26m-pose.pt \
    --top-k 5
"""

import argparse
import json
import os
import re
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Regex helpers ─────────────────────────────────────────────────────────────

FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")
TRACKLET_RE = re.compile(r"tracklet_(.+)$")


def parse_frame_id(path: Path) -> Optional[int]:
    m = FRAME_RE.search(path.name)
    return int(m.group(1)) if m else None


def parse_tracklet_id(tracklet_dir: Path) -> Tuple[str, Optional[int]]:
    m = TRACKLET_RE.search(tracklet_dir.name)
    raw = m.group(1) if m else tracklet_dir.name
    try:
        as_int = int(float(raw))
    except Exception:
        as_int = None
    return raw, as_int


def clamp_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(1, min(w, int(x2)))
    y2 = max(1, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# Confusões clássicas OCR em camisolas desportivas
_OCR_FIXES: Dict[str, str] = {
    "I": "1", "i": "1", "l": "1", "|": "1",
    "O": "0", "o": "0",
    "S": "5", "s": "5",
    "B": "8",
    "G": "6", "g": "6",
    "Z": "2", "z": "2",
    "T": "7",
    "A": "4",
    "q": "9", "Q": "9",
}

# Regex final aceite: exactamente 1 ou 2 dígitos
_JERSEY_RE = re.compile(r"^\d{1,2}$")
# Range válido de número de camisola
_JERSEY_MIN, _JERSEY_MAX = 1, 99


def extract_digits(text: str) -> Optional[str]:
    r"""
    Extract and validate jersey number from OCR text.

    Steps:
      1. Strip whitespace
      2. Apply OCR confusion corrections (I->1, O->0, S->5, ...)
      3. Keep only alnum chars
      4. Require high digit ratio (>=85%) to reject mixed strings
      5. Keep at most 2 digits
      6. Validate with regex ^\d{1,2}$
      7. Validate range 1..99
    """
    if not text:
        return None

    text = text.strip()

    # Apply OCR corrections char by char
    corrected = "".join(_OCR_FIXES.get(c, c) for c in text)

    # Keep only alnum
    cleaned = "".join(c for c in corrected if c.isalnum())
    if not cleaned:
        return None

    # Digit ratio check — reject "I9", "O1B", etc.
    digits = "".join(c for c in cleaned if c.isdigit())
    if not digits:
        return None
    if len(digits) / len(cleaned) < 0.85:
        return None

    # Keep at most 2 digits
    candidate = digits[:2]

    # Regex validation
    if not _JERSEY_RE.match(candidate):
        return None

    # Range validation
    try:
        n = int(candidate)
    except ValueError:
        return None
    if not (_JERSEY_MIN <= n <= _JERSEY_MAX):
        return None

    return candidate


# ─── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class PoseROICfg:
    kp_conf_min: float = 0.20
    face_conf_min: float = 0.30   # mais sensível: qualquer indício de face = frontal
    pad_x: float = 0.55           # alargado para capturar melhor o número nas costas
    pad_y_top: float = 0.10
    roi_height_frac: float = 0.90  # aumentado: captura mais área do tronco em Y


@dataclass
class LegibilityWeights:
    """Weights for the composite legibility score. Must sum to 1."""
    w_back: float = 0.35     # aumentado: face/costas é o critério mais importante
    w_size: float = 0.15
    w_blur: float = 0.20
    w_contrast: float = 0.10
    w_occ: float = 0.15     # aumentado: oclusão do número é crítica
    w_yaw: float = 0.05

    # Blur thresholds (Laplacian variance)
    blur_low: float = 20.0
    blur_high: float = 200.0

    # Contrast thresholds (std of gray)
    contrast_low: float = 15.0
    contrast_high: float = 80.0

    # Torso height minimum (px) for full size score
    torso_h_min: float = 30.0
    torso_h_full: float = 80.0

    # Yaw: shoulder width ratio threshold
    yaw_min_ratio: float = 0.30   # below this → nearly sideways → S_yaw ~ 0
    yaw_max_ratio: float = 0.85   # above this → nearly frontal-back → S_yaw ~ 1


@dataclass
class PipelineCfg:
    # legibility thresholds
    s_leg_ignore: float = 0.45
    s_leg_low_prio: float = 0.65
    s_leg_premium: float = 0.80

    # top-K frames per tracklet to send to OCR
    top_k: int = 5

    # OCR
    min_ocr_conf: float = 0.70
    final_ocr_conf: float = 0.80

    # OCR engine selection
    # use_paddle=True  → PaddleOCR (fast, frame-level), Qwen only as fallback
    # use_paddle=False → Qwen2.5-VL directly on every top-K frame (slower, mais preciso)
    use_paddle: bool = False

    # fallback (só relevante quando use_paddle=True)
    use_qwen_fallback: bool = False
    qwen_fallback_threshold: float = 0.50   # tracklet score below this → Qwen

    # preprocessing
    ocr_upscale: int = 2   # resize factor before OCR (1 = no resize)

    pose_conf: float = 0.25

    leg_weights: LegibilityWeights = field(default_factory=LegibilityWeights)
    pose_roi: PoseROICfg = field(default_factory=PoseROICfg)


# ─── COCO-17 keypoint indices ──────────────────────────────────────────────────

NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SH, R_SH = 5, 6
L_ELB, R_ELB = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12


# ─── Sub-scores ───────────────────────────────────────────────────────────────

def score_back(
    kxy: np.ndarray,
    kcf: np.ndarray,
    cfg: PoseROICfg,
    face_conf_min: float,
    kp_conf_min: float,
) -> float:
    """
    S_back: continuous 0..1.
    High when both shoulders visible, no face, torso approximately symmetric.

    Logic:
      - base quality from shoulder + hip visibility
      - penalty for face visible
      - penalty for torso asymmetry (one shoulder significantly lower confidence)
      - small penalty for tiny torso
    """
    def ok(i: int, th: float) -> bool:
        return float(kcf[i]) >= th

    sh_ok_l = ok(L_SH, kp_conf_min)
    sh_ok_r = ok(R_SH, kp_conf_min)
    hip_ok_l = ok(L_HIP, kp_conf_min)
    hip_ok_r = ok(R_HIP, kp_conf_min)

    shoulders_ok = sh_ok_l and sh_ok_r
    hips_ok = hip_ok_l and hip_ok_r

    # base quality
    if not shoulders_ok:
        return 0.0
    quality = 1.0 if hips_ok else 0.6

    # face visibility penalty (continuous)
    face_vis = float(
        ok(NOSE, face_conf_min) or ok(L_EYE, face_conf_min) or ok(R_EYE, face_conf_min)
    )
    s = (1.0 - face_vis) * quality

    # torso symmetry: both shoulder confs should be similar
    lsh_c = float(kcf[L_SH])
    rsh_c = float(kcf[R_SH])
    sh_min = min(lsh_c, rsh_c)
    sh_max = max(lsh_c, rsh_c)
    if sh_max > 0:
        asym = 1.0 - (sh_min / sh_max)   # 0 = symmetric, 1 = fully asymmetric
        s *= (1.0 - 0.5 * asym)

    return float(np.clip(s, 0.0, 1.0))


def score_size(
    kxy: np.ndarray,
    kcf: np.ndarray,
    kp_conf_min: float,
    torso_h_min: float,
    torso_h_full: float,
) -> float:
    """
    S_size: normalized torso height.
    Uses shoulder-to-hip distance as torso height proxy.
    """
    sh_ok = float(kcf[L_SH]) >= kp_conf_min and float(kcf[R_SH]) >= kp_conf_min
    hip_ok = float(kcf[L_HIP]) >= kp_conf_min and float(kcf[R_HIP]) >= kp_conf_min

    if not (sh_ok and hip_ok):
        # fallback: use just shoulder distance as width proxy
        if sh_ok:
            w = float(np.linalg.norm(kxy[L_SH] - kxy[R_SH]))
            return float(np.clip((w - torso_h_min) / max(1.0, torso_h_full - torso_h_min), 0.0, 1.0))
        return 0.0

    sh_mid = (kxy[L_SH] + kxy[R_SH]) / 2.0
    hip_mid = (kxy[L_HIP] + kxy[R_HIP]) / 2.0
    h = float(np.linalg.norm(sh_mid - hip_mid))
    return float(np.clip((h - torso_h_min) / max(1.0, torso_h_full - torso_h_min), 0.0, 1.0))


def score_blur(crop_bgr: np.ndarray, blur_low: float, blur_high: float) -> float:
    """
    S_blur: Laplacian variance of the crop.
    Higher variance → sharper image → higher score.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip((lap_var - blur_low) / max(1.0, blur_high - blur_low), 0.0, 1.0))


def score_contrast(crop_bgr: np.ndarray, contrast_low: float, contrast_high: float) -> float:
    """
    S_contrast: std of gray intensities in the crop.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    std = float(gray.std())
    return float(np.clip((std - contrast_low) / max(1.0, contrast_high - contrast_low), 0.0, 1.0))


def score_occlusion(
    kxy: np.ndarray,
    kcf: np.ndarray,
    kp_conf_min: float,
) -> float:
    """
    S_occ: 1 - occlusion estimate.

    Uses:
      - asymmetry of shoulder + hip keypoint confidences
      - wrists crossing into torso bounding box (arm occlusion)
    """
    sh_ok = float(kcf[L_SH]) >= kp_conf_min and float(kcf[R_SH]) >= kp_conf_min
    if not sh_ok:
        return 0.5   # can't tell

    lsh_c = float(kcf[L_SH])
    rsh_c = float(kcf[R_SH])
    lhip_c = float(kcf[L_HIP])
    rhip_c = float(kcf[R_HIP])

    # shoulder asymmetry penalty
    sh_asym = abs(lsh_c - rsh_c) / max(lsh_c + rsh_c, 1e-6)

    # hip asymmetry penalty (if hips detected)
    hip_asym = 0.0
    if lhip_c > kp_conf_min or rhip_c > kp_conf_min:
        hip_asym = abs(lhip_c - rhip_c) / max(lhip_c + rhip_c, 1e-6)

    # wrist crossing torso (arm occluding back number)
    wrist_penalty = 0.0
    torso_x_min = min(kxy[L_SH][0], kxy[R_SH][0])
    torso_x_max = max(kxy[L_SH][0], kxy[R_SH][0])
    for wi in [L_WR, R_WR]:
        if float(kcf[wi]) >= kp_conf_min:
            wx = float(kxy[wi][0])
            if torso_x_min <= wx <= torso_x_max:
                wrist_penalty += 0.20   # each wrist in torso zone adds penalty

    occ_estimate = 0.5 * sh_asym + 0.3 * hip_asym + min(wrist_penalty, 0.4)
    return float(np.clip(1.0 - occ_estimate, 0.0, 1.0))


def score_yaw(
    kxy: np.ndarray,
    kcf: np.ndarray,
    kp_conf_min: float,
    yaw_min_ratio: float,
    yaw_max_ratio: float,
) -> float:
    """
    S_yaw: how "squarely back-facing" the player is (not sideways).

    Estimate via ratio of observed shoulder width to expected shoulder width.
    When player faces squarely away, shoulder width is maximal.
    When sideways, width collapses and one shoulder may disappear.

    Uses the ratio: min_conf / max_conf of the two shoulders as a proxy
    for how parallel the back is to the camera plane.
    Combined with the geometric shoulder span.
    """
    lsh_c = float(kcf[L_SH])
    rsh_c = float(kcf[R_SH])

    if lsh_c < kp_conf_min and rsh_c < kp_conf_min:
        return 0.0

    # confidence symmetry ratio (1 = symmetric = square-on, 0 = one side invisible)
    sh_sum = lsh_c + rsh_c
    sh_min = min(lsh_c, rsh_c)
    conf_ratio = (2.0 * sh_min / sh_sum) if sh_sum > 0 else 0.0  # 0..1, 1 = symmetric

    # geometric: shoulder span vs image width proxy
    if lsh_c >= kp_conf_min and rsh_c >= kp_conf_min:
        sh_span = float(np.linalg.norm(kxy[L_SH] - kxy[R_SH]))
    else:
        sh_span = 0.0

    # normalise span by a reference (rough: shoulder span of 40px is minimum useful)
    span_score = float(np.clip((sh_span - 15.0) / max(1.0, 60.0 - 15.0), 0.0, 1.0))

    # combine: confidence ratio weighted more
    raw = 0.6 * conf_ratio + 0.4 * span_score

    # remap through yaw thresholds
    return float(np.clip((raw - yaw_min_ratio) / max(1e-6, yaw_max_ratio - yaw_min_ratio), 0.0, 1.0))


def compute_legibility_score(
    kxy: np.ndarray,
    kcf: np.ndarray,
    crop_bgr: Optional[np.ndarray],
    cfg: PipelineCfg,
) -> Dict[str, float]:
    """
    Compute all 6 sub-scores and composite S_leg.
    Returns dict with all sub-scores + s_leg.

    Hard gate: se s_back < 0.30 (face visível ou ombros muito assimétricos),
    o frame é rejeitado imediatamente (s_leg = 0) sem entrar no score composto.
    Isto evita processar jogadores de frente ou de lado.
    """
    w = cfg.leg_weights
    p = cfg.pose_roi

    s_back = score_back(kxy, kcf, p, p.face_conf_min, p.kp_conf_min)

    # ── Hard gate: rejeitar frames que claramente não estão de costas ──────────
    BACK_HARD_THRESHOLD = 0.30
    if s_back < BACK_HARD_THRESHOLD:
        return {
            "s_back": round(s_back, 4),
            "s_size": 0.0,
            "s_blur": 0.0,
            "s_contrast": 0.0,
            "s_occ": 0.0,
            "s_yaw": 0.0,
            "s_leg": 0.0,
        }

    s_size = score_size(kxy, kcf, p.kp_conf_min, w.torso_h_min, w.torso_h_full)
    s_occ = score_occlusion(kxy, kcf, p.kp_conf_min)
    s_yaw = score_yaw(kxy, kcf, p.kp_conf_min, w.yaw_min_ratio, w.yaw_max_ratio)

    if crop_bgr is not None and crop_bgr.size > 0:
        s_blur = score_blur(crop_bgr, w.blur_low, w.blur_high)
        s_contrast = score_contrast(crop_bgr, w.contrast_low, w.contrast_high)
    else:
        s_blur = 0.0
        s_contrast = 0.0

    s_leg = (
        w.w_back * s_back
        + w.w_size * s_size
        + w.w_blur * s_blur
        + w.w_contrast * s_contrast
        + w.w_occ * s_occ
        + w.w_yaw * s_yaw
    )

    return {
        "s_back": round(s_back, 4),
        "s_size": round(s_size, 4),
        "s_blur": round(s_blur, 4),
        "s_contrast": round(s_contrast, 4),
        "s_occ": round(s_occ, 4),
        "s_yaw": round(s_yaw, 4),
        "s_leg": round(float(np.clip(s_leg, 0.0, 1.0)), 4),
    }


def leg_tier(s_leg: float, cfg: PipelineCfg) -> str:
    """Classify S_leg into a named tier."""
    if s_leg < cfg.s_leg_ignore:
        return "ignore"
    elif s_leg < cfg.s_leg_low_prio:
        return "low_prio"
    elif s_leg < cfg.s_leg_premium:
        return "normal"
    else:
        return "premium"


# ─── Pose ROI (same logic as v1, kept here for self-containment) ───────────────

def estimate_back_roi(
    kxy: np.ndarray,
    kcf: np.ndarray,
    cfg: PoseROICfg,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute back ROI from COCO-17 keypoints.
    Returns (x1, y1, x2, y2) in image coordinates, or None.
    """
    def ok(i: int) -> bool:
        return float(kcf[i]) >= cfg.kp_conf_min

    sh_ok = ok(L_SH) and ok(R_SH)
    hip_ok = ok(L_HIP) and ok(R_HIP)

    if not sh_ok:
        return None

    lsh, rsh = kxy[L_SH], kxy[R_SH]
    x1 = min(lsh[0], rsh[0])
    x2 = max(lsh[0], rsh[0])
    y_sh = min(lsh[1], rsh[1])

    if hip_ok:
        lhip, rhip = kxy[L_HIP], kxy[R_HIP]
        y_hip = max(lhip[1], rhip[1])
    else:
        # estimate hip from shoulder span
        sh_w = max(1.0, x2 - x1)
        y_hip = y_sh + sh_w * 1.4

    torso_h = max(1.0, y_hip - y_sh)
    y1 = y_sh - cfg.pad_y_top * torso_h
    y2 = y_sh + cfg.roi_height_frac * torso_h
    pad_x = cfg.pad_x * max(1.0, x2 - x1)

    return (
        int(np.floor(x1 - pad_x)),
        int(np.floor(y1)),
        int(np.ceil(x2 + pad_x)),
        int(np.ceil(y2)),
    )


def pick_best_person(result) -> Optional[int]:
    """Pick person index with highest box confidence."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    conf = result.boxes.conf
    if conf is None or len(conf) == 0:
        return 0
    return int(conf.argmax().item())


# ─── OCR helpers ──────────────────────────────────────────────────────────────

def parse_paddle_rec_only(ocr_res: Any) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    if not ocr_res or not isinstance(ocr_res, list) or len(ocr_res) == 0:
        return out
    try:
        for item in ocr_res[0]:
            txt, conf = item
            out.append((str(txt), float(conf)))
    except Exception:
        return out
    return out


def preprocess_for_ocr(crop_bgr: np.ndarray, upscale: int = 2) -> np.ndarray:
    """
    Preprocess crop for OCR:
      1. Upscale (bilinear)
      2. Mild CLAHE for contrast normalisation
      3. Light unsharp mask (sharpen)
    """
    h, w = crop_bgr.shape[:2]
    if upscale > 1:
        img = cv2.resize(crop_bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
    else:
        img = crop_bgr.copy()

    # CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Unsharp mask
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    img = cv2.addWeighted(img, 1.4, blur, -0.4, 0)

    return img


# ─── Qwen2.5-VL fallback stub ─────────────────────────────────────────────────

def _load_qwen_model(model_path: str):
    """
    Load Qwen2.5-VL model + processor. Cached as module-level globals
    so we only pay the load cost once per process.
    Returns (model, processor) or (None, None) on failure.
    """
    global _QWEN_MODEL, _QWEN_PROCESSOR, _QWEN_LOADED_PATH
    if "_QWEN_MODEL" not in globals():
        _QWEN_MODEL = None
        _QWEN_PROCESSOR = None
        _QWEN_LOADED_PATH = None

    _FAILED_SENTINEL = "__failed__"
    if _QWEN_LOADED_PATH == model_path:
        # either cached or previously failed — don't retry
        return _QWEN_MODEL, _QWEN_PROCESSOR

    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        use_gpu = torch.cuda.is_available()

        if not use_gpu:
            warnings.warn(
                "[Qwen] No CUDA GPU available. Qwen2.5-VL requires a GPU for 4-bit quantization. "
                "Skipping Qwen fallback. Update your NVIDIA driver to enable GPU support."
            )
            return None, None

        print(f"[Qwen] Loading {model_path} (gpu=True, 4-bit)...")

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        kwargs = {"device_map": "auto", "quantization_config": bnb}

        proc = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
        model.eval()

        _QWEN_MODEL = model
        _QWEN_PROCESSOR = proc
        _QWEN_LOADED_PATH = model_path
        print("[Qwen] Model ready.")
        return model, proc
    except Exception as e:
        warnings.warn(f"[Qwen] Failed to load model: {e}")
        _QWEN_LOADED_PATH = model_path   # mark as failed so we don't retry
        return None, None


def unload_qwen_from_vram():
    """Release Qwen from VRAM after a video is processed.
    Forces a full reload on the next video — adds ~5s overhead but prevents OOM
    when calibration/pitch models run between videos.
    """
    global _QWEN_MODEL, _QWEN_PROCESSOR, _QWEN_LOADED_PATH
    import gc, torch
    try:
        if "_QWEN_MODEL" in globals() and _QWEN_MODEL is not None:
            del _QWEN_MODEL
        if "_QWEN_PROCESSOR" in globals() and _QWEN_PROCESSOR is not None:
            del _QWEN_PROCESSOR
        _QWEN_MODEL = None
        _QWEN_PROCESSOR = None
        _QWEN_LOADED_PATH = None   # allow reload next video
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[Qwen] Model unloaded from VRAM.", flush=True)
    except Exception as e:
        print(f"[Qwen] unload warning: {e}", flush=True)


def qwen_ocr_fallback(crop_bgr: np.ndarray, model_path: str) -> Optional[Tuple[str, float]]:
    """
    Qwen2.5-VL fallback OCR for jersey number recognition.

    Called when PaddleOCR fails, has low confidence, or tracklet has conflicts.
    Returns (jersey_number_str, confidence) or None.

    Confidence is heuristic:
      - 0.90 if reply is clean 1-2 digits
      - 0.70 if reply needed post-processing
      - None if no valid number found
    """
    import torch

    model, processor = _load_qwen_model(model_path)
    if model is None or processor is None:
        return None

    try:
        from qwen_vl_utils import process_vision_info

        # Convert BGR crop to RGB PIL image
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        prompt = (
        "Analyze this cropped image of the back of a sports jersey. "
        "Identify the jersey number printed on it.\n"
        "RULES:\n"
        "1. Output ONLY the number as 1 or 2 digits (e.g., 7, 10, 28).\n"
        "2. If two digits are present, output them together as a single number.\n"
        "3. If partially occluded or ambiguous, output ONLY the digits you are highly confident about. "
        "If no digit is clear, output exactly: none\n"
        "4. Do NOT add explanations, punctuation, spaces, or extra words. Only digits or 'none'.\n"
        "5. Pay attention to common jersey font confusions: 8/3, 6/5, 9/4, 1/7.\n"
        "Reply strictly following these rules."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device = next(model.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=6,    # números de camisola têm no máx 2 dígitos
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.1,
            )

        # Decode only the newly generated tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        reply = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        # Parse reply
        if reply.lower() in ("none", "n/a", "", "-", "?"):
            return None

        num = extract_digits(reply)
        if num is None:
            # Try harder: strip all non-digits
            digits_only = "".join(c for c in reply if c.isdigit())[:2]
            if digits_only:
                try:
                    n = int(digits_only)
                    if 1 <= n <= 99:
                        return (digits_only, 0.70)
                except ValueError:
                    pass
            return None

        # Clean reply (no extra chars) → higher confidence
        conf = 0.90 if reply.strip() == num else 0.75
        return (num, conf)

    except Exception as e:
        warnings.warn(f"[Qwen] Inference error: {e}")
        return None


# ─── Robust Qwen 3B output parser ────────────────────────────────────────────

def _clean_qwen_output(raw: str) -> Optional[str]:
    """
    Parser robusto para output do Qwen2.5-VL-3B.
    Lida com markdown (**10**), aspas ("28"), texto conversacional,
    dígitos separados por espaço ("2 8"), etc.
    Devolve string de 1-2 dígitos ou None.
    """
    if not raw:
        return None

    s = raw.strip().lower()

    # Rejeitar ruído conversacional
    if any(kw in s for kw in ["none", "unknown", "can't", "cannot", "unable",
                               "not visible", "not clear", "unclear", "n/a"]):
        return None

    # Remover markdown, aspas, parênteses, backticks
    s = re.sub(r'[*_"\'\[\]()`{}]', '', s)
    # Colapsar espaços (para "2 8" → "28")
    s = re.sub(r'\s+', '', s)

    # Extrair apenas dígitos (máx 2)
    digits = re.findall(r'\d', s)
    if not digits:
        return None

    candidate = "".join(digits[:2])
    try:
        n = int(candidate)
        return candidate if 1 <= n <= 99 else None
    except ValueError:
        return None


# ─── Batch Qwen inference ─────────────────────────────────────────────────────

# ─── Batch multi-tracklet inference ──────────────────────────────────────────

def qwen_ocr_multi_tracklet(
    tracklet_crops: Dict[Any, List[np.ndarray]],
    model_path: str,
    max_total_crops: int = 50,
) -> Dict[Any, Optional[Tuple[str, float]]]:
    """
    Batch inference: ONE Qwen call for N tracklets simultaneously.

    Instead of N serial calls (one per tracklet), all tracklets are packed
    into a single call with grouped images.  The speedup scales roughly
    linearly with the number of tracklets batched.

    Args:
        tracklet_crops: {tid: [crop_bgr, ...]}  — crops already in BGR
        model_path:     HF id or local path for Qwen2.5-VL
        max_total_crops: hard cap on total images sent (VRAM guard)

    Returns:
        {tid: (number_str, confidence) or None}
    """
    import torch

    if not tracklet_crops:
        return {}

    model, processor = _load_qwen_model(model_path)
    if model is None or processor is None:
        return {tid: None for tid in tracklet_crops}

    try:
        from qwen_vl_utils import process_vision_info
        from PIL import Image as PILImage

        # ── Build ordered list to maintain tid→group index mapping ──────────
        tids_ordered = list(tracklet_crops.keys())
        n_groups = len(tids_ordered)

        # Enforce VRAM cap: trim crops per group proportionally
        # Sort by sharpness descending so the sharpest crops are sent, not just the first ones
        crops_per_group = max(1, max_total_crops // n_groups)

        def _sharpness(crop_bgr: np.ndarray) -> float:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

        groups: List[List[np.ndarray]] = [
            sorted(tracklet_crops[tid], key=_sharpness, reverse=True)[:crops_per_group]
            for tid in tids_ordered
        ]

        # ── Build content: interleave group labels + images ──────────────────
        content: List[Dict] = []
        for g_idx, (tid, crops) in enumerate(zip(tids_ordered, groups)):
            content.append({"type": "text",
                            "text": f"Group {g_idx} ({len(crops)} frames):"})
            for crop_bgr in crops:
                rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                content.append({"type": "image",
                                 "image": PILImage.fromarray(rgb)})

        # ── Prompt ───────────────────────────────────────────────────────────
        lines_fmt = "\n".join(f"group{i}: <number or none>"
                              for i in range(n_groups))
        prompt = (
            f"I'm showing you {n_groups} groups of jersey images. "
            "Each group shows the BACK of the same sports player from different video frames.\n"
            "Task: for each group, identify the jersey number printed on the back.\n\n"
            f"Output format — exactly {n_groups} lines, no other text:\n"
            f"{lines_fmt}\n\n"
            "Rules:\n"
            "- Output ONLY 1-2 digits (e.g. 7, 28) or the word 'none'.\n"
            "- Always output the COMPLETE number — never drop a digit. "
            "If you see '1' followed by '4', output '14', not '4'.\n"
            "- Jersey numbers are 1–99. A single digit like '4' is valid, but first check "
            "if there is a leading digit partially visible at the edge (e.g. '14', '24', '34').\n"
            "- Only output a number if you are CONFIDENT you can read every digit clearly. "
            "If the images are too small, blurry, at a bad angle, or you would be guessing, "
            "output 'none' instead — a wrong guess is worse than admitting you can't read it.\n"
            "- NO markdown, NO quotes, NO explanations."
        )
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a jersey number extractor. "
                    "Your job is to read the COMPLETE number on the back of a sports jersey. "
                    "Always output all visible digits together. Never truncate a number."
                ),
            },
            {"role": "user", "content": content},
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        # Sync any pending YOLO/GPU ops before Qwen generate to avoid corrupted state.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max(20, n_groups * 12),
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        reply = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        # ── Parse "group0: 7\ngroup1: none\ngroup2: 28\n..." ─────────────────
        print(f"[Qwen] {n_groups} groups reply={reply!r}", flush=True)
        results: Dict[Any, Optional[Tuple[str, float]]] = {}
        parsed: Dict[int, Optional[str]] = {}
        for line in reply.splitlines():
            m = re.match(r'group\s*(\d+)\s*[:\-]\s*(.+)', line.strip(), re.IGNORECASE)
            if m:
                g_idx = int(m.group(1))
                if 0 <= g_idx < n_groups:
                    parsed[g_idx] = _clean_qwen_output(m.group(2).strip())

        for g_idx, tid in enumerate(tids_ordered):
            num = parsed.get(g_idx)
            results[tid] = (num, 0.90) if num else None

        return results

    except Exception as e:
        warnings.warn(f"[Qwen-multi-tracklet] Inference error: {e}")
        return {tid: None for tid in tracklet_crops}


def qwen_ocr_batch(
    crops_bgr: List[np.ndarray],
    model_path: str,
) -> List[Optional[Tuple[str, float]]]:
    """
    Mantido para compatibilidade. Envia cada crop individualmente.
    Preferir qwen_ocr_tracklet() quando os crops são do mesmo tracklet.
    """
    import torch

    if not crops_bgr:
        return []

    model, processor = _load_qwen_model(model_path)
    if model is None or processor is None:
        return [None] * len(crops_bgr)

    try:
        from qwen_vl_utils import process_vision_info
        from PIL import Image as PILImage

        results: List[Optional[Tuple[str, float]]] = []

        for crop_bgr in crops_bgr:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)

            prompt = (
                "Task: Read the jersey number on the back.\n"
                "Output format: ONLY 1-2 digits or the exact word 'none'.\n"
                "Rules:\n"
                "- If you see 2 digits, write them together (e.g., 28).\n"
                "- If partially hidden, write only the digits you are 100% sure of.\n"
                "- If unclear, write: none\n"
                "- NO quotes, NO spaces, NO markdown, NO explanations.\n"
                "Examples:\n"
                "Clear '7' → 7\n"
                "Clear '28' → 28\n"
                "Blurry → none"
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are a strict digit extractor. Output only numbers or 'none'.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
            device = next(model.parameters()).device
            inputs = inputs.to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    repetition_penalty=1.1,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            reply = processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0].strip()

            num = _clean_qwen_output(reply)
            results.append((num, 0.85) if num else None)

        return results

    except Exception as e:
        warnings.warn(f"[Qwen-batch] Inference error: {e}")
        return [None] * len(crops_bgr)


# ─── Tracklet-level aggregation ───────────────────────────────────────────────

def aggregate_tracklet(
    frames: List[Dict[str, Any]],
    cfg: PipelineCfg,
) -> Dict[str, Any]:
    """
    Confidence-weighted temporal aggregation for a tracklet.

    score(n) = Σ_i [ s_leg_i * ocr_conf_i ]  for all frames i where best_num == n

    Returns:
      - jersey_number: winning number (or None)
      - tracklet_score: normalised score of the winner
      - conflict: True if top-2 candidates are within 20% of each other
      - needs_fallback: True if should go to Qwen
      - all_candidates: sorted list of (number, score)
    """
    # collect only frames that produced a valid OCR hit
    hits = [f for f in frames if f.get("ocr_best_num") is not None and f.get("run_ocr")]
    if not hits:
        return {
            "jersey_number": None,
            "tracklet_score": 0.0,
            "conflict": False,
            "needs_fallback": True,
            "all_candidates": [],
            "n_hits": 0,
            "n_frames": len(frames),
            "n_ocr_frames": sum(1 for f in frames if f.get("run_ocr")),
        }

    # weighted vote
    vote: Dict[str, float] = {}
    for f in hits:
        n = str(f["ocr_best_num"])
        s_leg = float(f.get("s_leg", 0.5))
        oc = float(f.get("ocr_best_conf", 0.0))
        vote[n] = vote.get(n, 0.0) + s_leg * oc

    total = sum(vote.values())
    sorted_cands = sorted(vote.items(), key=lambda x: x[1], reverse=True)
    normalised = [(n, s / max(total, 1e-9)) for n, s in sorted_cands]

    winner_num, winner_score = normalised[0]

    # conflict detection
    conflict = False
    if len(normalised) >= 2:
        second_score = normalised[1][1]
        if second_score > 0 and (winner_score - second_score) / max(winner_score, 1e-9) < 0.20:
            conflict = True

    needs_fallback = (
        winner_score < cfg.qwen_fallback_threshold
        or conflict
        or len(hits) < 2
    )

    return {
        "jersey_number": winner_num,
        "tracklet_score": round(winner_score, 4),
        "conflict": conflict,
        "needs_fallback": needs_fallback,
        "all_candidates": normalised,
        "n_hits": len(hits),
        "n_frames": len(frames),
        "n_ocr_frames": sum(1 for f in frames if f.get("run_ocr")),
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def save_plots(det: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if det.empty:
        return

    ocr_rows = det.loc[det.get("run_ocr", pd.Series(False, index=det.index)).fillna(False).astype(bool)].copy()
    ocr_rows = ocr_rows.loc[ocr_rows["ocr_best_num"].notna()]
    if ocr_rows.empty:
        return

    for (tid_raw,), g in ocr_rows.groupby(["tracklet_id_raw"]):
        g = g.sort_values("frame_id")
        nums = g["ocr_best_num"].astype(str).tolist()
        confs = g["ocr_best_conf"].astype(float).tolist()
        frames = g["frame_id"].astype(float).tolist()
        s_legs = g["s_leg"].astype(float).tolist() if "s_leg" in g.columns else [0.5] * len(g)

        # Histogram
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        pd.Series(nums).value_counts().plot(kind="bar", ax=axes[0])
        axes[0].set_title(f"Tracklet {tid_raw} — detected numbers")
        axes[0].set_ylabel("count")

        # S_leg distribution
        axes[1].hist(s_legs, bins=20, range=(0, 1), color="steelblue")
        axes[1].set_title(f"Tracklet {tid_raw} — S_leg distribution")
        axes[1].set_xlabel("S_leg")
        axes[1].set_ylabel("frames")
        plt.tight_layout()
        plt.savefig(plots_dir / f"tracklet_{tid_raw}_hist.png", dpi=150)
        plt.close()

        # Timeline coloured by S_leg
        uniq = sorted(set(nums))
        ymap = {n: i for i, n in enumerate(uniq)}
        ys = [ymap[n] for n in nums]
        fig, ax = plt.subplots(figsize=(10, 3))
        sc = ax.scatter(frames, ys, c=s_legs, cmap="viridis", s=40, vmin=0.0, vmax=1.0)
        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels(list(ymap.keys()))
        plt.colorbar(sc, ax=ax, label="S_leg")
        ax.set_title(f"Tracklet {tid_raw} — timeline (colour=S_leg)")
        ax.set_xlabel("frame_id")
        plt.tight_layout()
        plt.savefig(plots_dir / f"tracklet_{tid_raw}_timeline.png", dpi=150)
        plt.close()


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    tracklets_dir: Path,
    out_dir: Path,
    samples_per_tracklet: int,
    max_tracklets: Optional[int],
    pose_weights: str,
    ocr_lang: str,
    ocr_use_gpu: bool,
    cfg: PipelineCfg,
    qwen_model_path: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"[JNR-v2] Loading pose model: {pose_weights}")
    # Resolver para path absoluto se relativo (evita erros de CWD no TrackLab)
    pose_weights_path = Path(pose_weights)
    if not pose_weights_path.is_absolute():
        # Procurar em locais conhecidos
        candidates = [
            Path("/home/joao/soccernet/weights") / pose_weights,
            Path("/home/joao/soccernet/sn-gamestate") / pose_weights,
            Path("/home/joao/soccernet") / pose_weights,
            pose_weights_path,
        ]
        for c in candidates:
            if c.exists():
                pose_weights_path = c
                break
    pose_model = YOLO(str(pose_weights_path))

    ocr = None
    if cfg.use_paddle:
        print(f"[JNR-v2] Loading PaddleOCR (lang={ocr_lang}, gpu={ocr_use_gpu})")
        from paddleocr import PaddleOCR  # lazy import — evita segfault quando não é necessário
        ocr = PaddleOCR(lang=ocr_lang, use_gpu=ocr_use_gpu, show_log=False, use_angle_cls=False)
    else:
        print(f"[JNR-v2] PaddleOCR DISABLED — using Qwen2.5-VL directly on top-K frames")
        # Pre-load Qwen DEPOIS de libertar o YOLO da GPU (6GB VRAM é insuficiente para ambos)
        # O YOLO é carregado novamente se necessário, mas as inferências YOLO são feitas antes do Qwen
        if qwen_model_path:
            import gc
            # Não carregamos ainda — o YOLO precisa de correr primeiro
            # O Qwen é carregado lazy dentro de qwen_ocr_batch(), após o YOLO terminar
            print(f"[JNR-v2] Qwen será carregado lazy após inferência YOLO (economiza VRAM)")

    tdirs = sorted([p for p in tracklets_dir.iterdir() if p.is_dir() and p.name.startswith("tracklet_")])
    if max_tracklets is not None:
        tdirs = tdirs[:max_tracklets]
    print(f"[JNR-v2] Processing {len(tdirs)} tracklets from {tracklets_dir}")

    # ── Per-frame rows ─────────────────────────────────────────────────────────
    frame_rows: List[Dict[str, Any]] = []
    _yolo_unloaded = False  # flag para descarregar YOLO da GPU antes do Qwen

    for tdir in tdirs:
        tracklet_raw, tracklet_id = parse_tracklet_id(tdir)
        imgs = sorted(tdir.glob("frame_*.jpg"))
        if not imgs:
            continue

        # uniform sample (initial broad sample to compute S_leg)
        if samples_per_tracklet > 0 and len(imgs) > samples_per_tracklet:
            idxs = np.linspace(0, len(imgs) - 1, samples_per_tracklet).round().astype(int)
            imgs_sel = [imgs[i] for i in idxs]
        else:
            imgs_sel = imgs

        tdbg = debug_dir / f"tracklet_{tracklet_raw}"
        tdbg.mkdir(parents=True, exist_ok=True)

        # ── Pass 1: compute S_leg for all frames ────────────────────────────
        pass1: List[Dict[str, Any]] = []

        for img_path in imgs_sel:
            frame_id = parse_frame_id(img_path)
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]

            pred = pose_model.predict(im, conf=cfg.pose_conf, verbose=False, workers=0)[0]
            if pred.keypoints is None or len(pred.keypoints.xy) == 0:
                pass1.append({
                    "img_path": img_path, "frame_id": frame_id, "im": im,
                    "pose_found": False, "s_leg": 0.0, "tier": "ignore",
                    "kxy": None, "kcf": None, "roi": None,
                })
                continue

            pi = pick_best_person(pred)
            if pi is None:
                pass1.append({
                    "img_path": img_path, "frame_id": frame_id, "im": im,
                    "pose_found": False, "s_leg": 0.0, "tier": "ignore",
                    "kxy": None, "kcf": None, "roi": None,
                })
                continue

            kxy = pred.keypoints.xy[pi].cpu().numpy()
            kcf = pred.keypoints.conf[pi].cpu().numpy()

            roi0 = estimate_back_roi(kxy, kcf, cfg.pose_roi)
            roi = clamp_xyxy(*roi0, w=w, h=h) if roi0 else None

            # extract crop for blur/contrast sub-scores
            crop_for_scores = None
            if roi:
                x1, y1, x2, y2 = roi
                crop_for_scores = im[y1:y2, x1:x2]

            scores = compute_legibility_score(kxy, kcf, crop_for_scores, cfg)
            tier = leg_tier(scores["s_leg"], cfg)

            pass1.append({
                "img_path": img_path, "frame_id": frame_id, "im": im,
                "pose_found": True, "kxy": kxy, "kcf": kcf, "roi": roi,
                **scores, "tier": tier,
            })

        # ── Top-K selection for OCR ─────────────────────────────────────────
        eligible = [f for f in pass1 if f["tier"] in ("normal", "premium")]
        eligible.sort(key=lambda f: f["s_leg"], reverse=True)
        top_k_set = set(id(f) for f in eligible[: cfg.top_k])

        # ── Pass 2: run OCR on top-K ────────────────────────────────────────
        # Pre-extract crops for all top-K frames (needed for Qwen batch)
        topk_entries = [e for e in pass1 if id(e) in top_k_set and e.get("roi") and e["im"] is not None]

        if not cfg.use_paddle and qwen_model_path and topk_entries:
            # ── Qwen BATCH: processar todos os top-K crops de uma vez ──────
            t_batch_start = time.perf_counter()
            topk_crops = []
            topk_rois_valid = []
            for e in topk_entries:
                x1, y1, x2, y2 = e["roi"]
                crop = e["im"][y1:y2, x1:x2]
                if crop is not None and crop.size > 0:
                    topk_crops.append(crop)
                    topk_rois_valid.append(True)
                else:
                    topk_crops.append(None)
                    topk_rois_valid.append(False)

            valid_crops = [c for c in topk_crops if c is not None]

            # ── Libertar YOLO da GPU antes do Qwen (economiza VRAM) ──────────
            if not _yolo_unloaded and valid_crops:
                import gc
                pose_model.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()
                _yolo_unloaded = True
                print("[JNR-v2] YOLO movido para CPU — VRAM libertada para Qwen")

            t_qwen_start = time.perf_counter()
            batch_results = qwen_ocr_batch(valid_crops, qwen_model_path) if valid_crops else []
            t_qwen_end = time.perf_counter()
            qwen_total_s = t_qwen_end - t_qwen_start
            qwen_per_crop_s = qwen_total_s / max(len(valid_crops), 1)
            print(f"  [Qwen-batch] tracklet={tracklet_raw} | {len(valid_crops)} crops | "
                  f"total={qwen_total_s:.2f}s | per_crop={qwen_per_crop_s:.2f}s")

            # Remontar resultados (incluindo None para crops inválidos)
            result_iter = iter(batch_results)
            qwen_results_by_entry: Dict[int, Optional[Tuple[str, float]]] = {}
            qwen_time_by_entry: Dict[int, float] = {}
            for e, valid in zip(topk_entries, topk_rois_valid):
                if valid:
                    qwen_results_by_entry[id(e)] = next(result_iter)
                    qwen_time_by_entry[id(e)] = qwen_per_crop_s
                else:
                    qwen_results_by_entry[id(e)] = None
                    qwen_time_by_entry[id(e)] = 0.0
        else:
            qwen_results_by_entry = {}
            qwen_time_by_entry = {}

        for entry in pass1:
            run_ocr = id(entry) in top_k_set
            roi = entry.get("roi")
            im = entry["im"]
            img_path = entry["img_path"]
            frame_id = entry["frame_id"]
            tracklet_scores = {k: entry.get(k, 0.0) for k in
                               ["s_back", "s_size", "s_blur", "s_contrast", "s_occ", "s_yaw", "s_leg"]}

            best_num: Optional[str] = None
            best_conf: float = 0.0
            ocr_raw: Any = None
            crop_path: Optional[str] = None
            roi_path: Optional[str] = None
            ocr_path: Optional[str] = None
            skip_reason = ""
            ocr_time_s: float = 0.0

            if not entry["pose_found"]:
                skip_reason = "no_pose"
            elif entry["tier"] == "ignore":
                skip_reason = f"s_leg_too_low({entry.get('s_leg', 0.0):.2f})"
            elif entry["tier"] == "low_prio" and id(entry) not in top_k_set:
                skip_reason = f"low_prio_not_in_topk"
            elif not run_ocr:
                skip_reason = "not_in_topk"

            if run_ocr and roi and im is not None:
                x1, y1, x2, y2 = roi

                # debug: ROI overlay
                roi_img = im.copy()
                cv2.rectangle(roi_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                roi_path = str(tdbg / f"{img_path.stem}_roi.jpg")
                cv2.imwrite(roi_path, roi_img)

                crop = im[y1:y2, x1:x2]
                if crop is not None and crop.size > 0:

                    if cfg.use_paddle:
                        # ── PaddleOCR path ──────────────────────────────────
                        t0 = time.perf_counter()
                        ocr_input = preprocess_for_ocr(crop, upscale=cfg.ocr_upscale)
                        res = ocr.ocr(ocr_input, det=False, cls=False)
                        ocr_time_s = time.perf_counter() - t0
                        cands = []
                        for txt, conf in parse_paddle_rec_only(res):
                            num = extract_digits(txt)
                            if num is None:
                                continue
                            cands.append((num, float(conf)))
                            if float(conf) > best_conf:
                                best_conf = float(conf)
                                best_num = num
                        ocr_raw = cands

                        if best_num is not None and best_conf < cfg.final_ocr_conf:
                            best_num = None
                            best_conf = 0.0

                        nr_tag = f"paddle{best_num}" if best_num else "paddleNone"
                        crop_path = str(tdbg / f"{img_path.stem}_crop_{nr_tag}.jpg")
                        cv2.imwrite(crop_path, crop)

                        try:
                            dbg_img = ocr_input.copy()
                            cv2.putText(
                                dbg_img,
                                f"{best_num} ({best_conf:.2f}) S={entry.get('s_leg', 0):.2f}",
                                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                            )
                            ocr_path = str(tdbg / f"{img_path.stem}_ocr_{nr_tag}.jpg")
                            cv2.imwrite(ocr_path, dbg_img)
                        except Exception:
                            pass

                    else:
                        # ── Qwen BATCH path — resultado já computado ────────
                        qwen_frame_result = qwen_results_by_entry.get(id(entry))
                        ocr_time_s = qwen_time_by_entry.get(id(entry), 0.0)
                        if qwen_frame_result is not None:
                            best_num, best_conf = qwen_frame_result
                        ocr_raw = [(best_num, best_conf)] if best_num else []

                        nr_tag = f"qwen{best_num}" if best_num else "qwenNone"
                        crop_path = str(tdbg / f"{img_path.stem}_crop_{nr_tag}.jpg")
                        cv2.imwrite(crop_path, crop)

                        try:
                            dbg_img = preprocess_for_ocr(crop, upscale=cfg.ocr_upscale)
                            cv2.putText(
                                dbg_img,
                                f"Q:{best_num} ({best_conf:.2f}) S={entry.get('s_leg', 0):.2f}",
                                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                            )
                            ocr_path = str(tdbg / f"{img_path.stem}_ocr_{nr_tag}.jpg")
                            cv2.imwrite(ocr_path, dbg_img)
                        except Exception:
                            pass

            frame_rows.append({
                "tracklet_id_raw": tracklet_raw,
                "tracklet_id": tracklet_id,
                "frame_id": frame_id,
                "img_path": str(img_path),
                "pose_found": entry["pose_found"],
                "tier": entry.get("tier", "ignore"),
                **tracklet_scores,
                "roi_x1": roi[0] if roi else None,
                "roi_y1": roi[1] if roi else None,
                "roi_x2": roi[2] if roi else None,
                "roi_y2": roi[3] if roi else None,
                "run_ocr": bool(run_ocr),
                "skip_reason": skip_reason if not run_ocr else None,
                "crop_path": crop_path,
                "debug_roi_path": roi_path,
                "debug_ocr_path": ocr_path,
                "ocr_best_num": best_num,
                "ocr_best_conf": float(best_conf),
                "ocr_raw": json.dumps(ocr_raw, ensure_ascii=False) if ocr_raw else None,
                "ocr_time_s": round(ocr_time_s, 4),
            })

    det = pd.DataFrame(frame_rows)

    # ── Tracklet-level aggregation ─────────────────────────────────────────────
    summary_rows: List[Dict[str, Any]] = []
    for (tid_raw, tid), g in det.groupby(["tracklet_id_raw", "tracklet_id"], dropna=False):
        frames_list = g.to_dict("records")
        agg = aggregate_tracklet(frames_list, cfg)

        # ── Qwen tracklet-level fallback (só quando use_paddle=True) ──────────
        # Quando use_paddle=False o Qwen já correu frame-a-frame; não há fallback adicional.
        qwen_result = None
        if cfg.use_paddle and agg["needs_fallback"] and cfg.use_qwen_fallback and qwen_model_path:
            # find best crop for Qwen
            top_frame = max(
                [f for f in frames_list if f.get("run_ocr") and f.get("crop_path")],
                key=lambda f: f.get("s_leg", 0),
                default=None,
            )
            if top_frame and top_frame.get("crop_path"):
                crop_bgr = cv2.imread(top_frame["crop_path"])
                if crop_bgr is not None:
                    qwen_result = qwen_ocr_fallback(crop_bgr, qwen_model_path)

        final_number = agg["jersey_number"]
        final_source = "qwen" if not cfg.use_paddle else "paddle"
        paddle_number = final_number  # guardar predição paddle antes de substituir
        if qwen_result is not None:
            final_number = qwen_result[0]
            final_source = "qwen"
            # O crop já tem "_paddleX" no nome; adicionar "_qwenY" para identificação visual
            # Resultado final: frame_XXXX_crop_paddle7_qwen9.jpg
            if top_frame and top_frame.get("crop_path"):
                old_path = Path(top_frame["crop_path"])
                if old_path.exists():
                    qwen_tag = f"qwen{final_number}" if final_number else "qwenNone"
                    new_name = old_path.stem + f"_{qwen_tag}" + old_path.suffix
                    new_path = old_path.parent / new_name
                    old_path.rename(new_path)
                    top_frame["crop_path"] = str(new_path)

        mean_s_leg = float(g["s_leg"].mean()) if "s_leg" in g.columns else 0.0
        max_s_leg = float(g["s_leg"].max()) if "s_leg" in g.columns else 0.0
        mean_ocr_time = float(g["ocr_time_s"].mean()) if "ocr_time_s" in g.columns else 0.0
        total_ocr_time = float(g["ocr_time_s"].sum()) if "ocr_time_s" in g.columns else 0.0

        summary_rows.append({
            "tracklet_id_raw": tid_raw,
            "tracklet_id": tid,
            "jersey_number": final_number,
            "jersey_source": final_source,
            "tracklet_score": agg["tracklet_score"],
            "conflict": agg["conflict"],
            "needs_fallback": agg["needs_fallback"],
            "all_candidates": json.dumps(agg["all_candidates"], ensure_ascii=False),
            "n_frames": agg["n_frames"],
            "n_ocr_frames": agg["n_ocr_frames"],
            "n_hits": agg["n_hits"],
            "mean_s_leg": round(mean_s_leg, 4),
            "max_s_leg": round(max_s_leg, 4),
            "mean_ocr_time_s": round(mean_ocr_time, 4),
            "total_ocr_time_s": round(total_ocr_time, 4),
        })

    summary = pd.DataFrame(summary_rows)

    # ── Save outputs ───────────────────────────────────────────────────────────
    det_path = out_dir / "detections_jnr_v2.parquet"
    sum_path = out_dir / "tracklets_jnr_v2_summary.parquet"
    det.to_parquet(det_path, index=False)
    summary.to_parquet(sum_path, index=False)

    print(f"[JNR-v2] Wrote: {det_path}")
    print(f"[JNR-v2] Wrote: {sum_path}")

    # also CSV for easy inspection
    summary.to_csv(out_dir / "tracklets_jnr_v2_summary.csv", index=False)
    print(f"[JNR-v2] Wrote: {out_dir / 'tracklets_jnr_v2_summary.csv'}")

    save_plots(det, out_dir)
    print(f"[JNR-v2] Plots: {out_dir / 'plots'}")
    print(f"[JNR-v2] Debug: {debug_dir}")

    # quick summary stats
    total = len(summary)
    with_number = int(summary["jersey_number"].notna().sum())
    conflicts = int(summary["conflict"].sum())
    fallbacks = int(summary["needs_fallback"].sum())
    print(f"\n[JNR-v2] Summary: {total} tracklets | "
          f"{with_number} with number ({100*with_number/max(total,1):.1f}%) | "
          f"{conflicts} conflicts | {fallbacks} need fallback")

    # timing stats
    if "ocr_time_s" in det.columns:
        ocr_frames = det[det["ocr_time_s"] > 0]
        if len(ocr_frames) > 0:
            total_ocr = ocr_frames["ocr_time_s"].sum()
            mean_ocr  = ocr_frames["ocr_time_s"].mean()
            n_ocr     = len(ocr_frames)
            engine    = "Qwen" if not cfg.use_paddle else "Paddle"
            print(f"[JNR-v2] {engine} timing: {n_ocr} crops | "
                  f"total={total_ocr:.1f}s | mean/crop={mean_ocr:.2f}s | "
                  f"throughput={n_ocr/max(total_ocr,0.001):.1f} crops/s")

    # per-tracklet timing CSV
    if "total_ocr_time_s" in summary.columns:
        timing_cols = ["tracklet_id_raw", "jersey_number", "n_ocr_frames",
                       "mean_ocr_time_s", "total_ocr_time_s"]
        timing_df = summary[timing_cols].copy()
        timing_path = out_dir / "tracklets_timing.csv"
        timing_df.to_csv(timing_path, index=False)
        print(f"[JNR-v2] Timing CSV: {timing_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="JNR v2: Legibility-score-guided Jersey Number Recognition"
    )

    # I/O
    ap.add_argument("--tracklets-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--samples-per-tracklet", type=int, default=50,
                    help="Max frames to sample per tracklet (0 = all)")
    ap.add_argument("--max-tracklets", type=int, default=None)

    # Pose
    ap.add_argument("--pose-weights", type=str, default="yolo26m-pose.pt")
    ap.add_argument("--pose-conf", type=float, default=0.25)

    # OCR
    ap.add_argument("--ocr-lang", type=str, default="en")
    ap.add_argument("--ocr-use-gpu", action="store_true", default=False)
    ap.add_argument("--min-ocr-conf", type=float, default=0.70)
    ap.add_argument("--final-ocr-conf", type=float, default=0.80)
    ap.add_argument("--ocr-upscale", type=int, default=2,
                    help="Upscale factor before OCR (1 = no resize)")

    # Legibility thresholds
    ap.add_argument("--s-leg-ignore", type=float, default=0.45)
    ap.add_argument("--s-leg-low-prio", type=float, default=0.65)
    ap.add_argument("--s-leg-premium", type=float, default=0.80)
    ap.add_argument("--top-k", type=int, default=5,
                    help="Top-K frames per tracklet to send to OCR")

    # Sub-score weights
    ap.add_argument("--w-back", type=float, default=0.25)
    ap.add_argument("--w-size", type=float, default=0.15)
    ap.add_argument("--w-blur", type=float, default=0.25)
    ap.add_argument("--w-contrast", type=float, default=0.15)
    ap.add_argument("--w-occ", type=float, default=0.10)
    ap.add_argument("--w-yaw", type=float, default=0.10)

    # Sub-score thresholds
    ap.add_argument("--blur-low", type=float, default=20.0)
    ap.add_argument("--blur-high", type=float, default=200.0)
    ap.add_argument("--contrast-low", type=float, default=15.0)
    ap.add_argument("--contrast-high", type=float, default=80.0)
    ap.add_argument("--torso-h-min", type=float, default=30.0)
    ap.add_argument("--torso-h-full", type=float, default=80.0)

    # Pose ROI
    ap.add_argument("--kp-conf-min", type=float, default=0.20)
    ap.add_argument("--face-conf-min", type=float, default=0.35)
    ap.add_argument("--pad-x", type=float, default=0.35)
    ap.add_argument("--pad-y-top", type=float, default=0.10)
    ap.add_argument("--roi-height-frac", type=float, default=0.75)

    # Aggregation
    ap.add_argument("--qwen-fallback-threshold", type=float, default=0.50)

    # OCR engine selection
    ap.add_argument("--use-paddle", action="store_true", default=False,
                    help="Use PaddleOCR instead of Qwen (Qwen is the default)")
    # Qwen fallback (só activo quando --use-paddle)
    ap.add_argument("--qwen-fallback", action="store_true", default=False,
                    help="Enable Qwen2.5-VL fallback after Paddle (requires --use-paddle --qwen-model-path)")
    ap.add_argument("--qwen-model-path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                    help="Path or HF id for Qwen2.5-VL model")

    return ap


def main() -> None:
    args = build_argparser().parse_args()

    leg_w = LegibilityWeights(
        w_back=args.w_back,
        w_size=args.w_size,
        w_blur=args.w_blur,
        w_contrast=args.w_contrast,
        w_occ=args.w_occ,
        w_yaw=args.w_yaw,
        blur_low=args.blur_low,
        blur_high=args.blur_high,
        contrast_low=args.contrast_low,
        contrast_high=args.contrast_high,
        torso_h_min=args.torso_h_min,
        torso_h_full=args.torso_h_full,
    )

    pose_roi = PoseROICfg(
        kp_conf_min=args.kp_conf_min,
        face_conf_min=args.face_conf_min,
        pad_x=args.pad_x,
        pad_y_top=args.pad_y_top,
        roi_height_frac=args.roi_height_frac,
    )

    cfg = PipelineCfg(
        s_leg_ignore=args.s_leg_ignore,
        s_leg_low_prio=args.s_leg_low_prio,
        s_leg_premium=args.s_leg_premium,
        top_k=args.top_k,
        min_ocr_conf=args.min_ocr_conf,
        final_ocr_conf=args.final_ocr_conf,
        ocr_upscale=args.ocr_upscale,
        use_paddle=args.use_paddle,
        use_qwen_fallback=args.qwen_fallback,
        qwen_fallback_threshold=args.qwen_fallback_threshold,
        pose_conf=args.pose_conf,
        leg_weights=leg_w,
        pose_roi=pose_roi,
    )

    run_pipeline(
        tracklets_dir=Path(args.tracklets_dir),
        out_dir=Path(args.out_dir),
        samples_per_tracklet=args.samples_per_tracklet,
        max_tracklets=args.max_tracklets,
        pose_weights=args.pose_weights,
        ocr_lang=args.ocr_lang,
        ocr_use_gpu=bool(args.ocr_use_gpu),
        cfg=cfg,
        qwen_model_path=args.qwen_model_path,
    )


if __name__ == "__main__":
    main()
