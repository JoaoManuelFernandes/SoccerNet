"""
tracklets_jnr_v2_module.py  (v5 — timing profiling)
=====================================================
DetectionLevelModule — mesma interface da baseline TrackletsOCR.
Motor: YOLO26m-pose + Qwen2.5-VL.

Timing logs: grep [JNR-v2 TIMING] no output para ver bottlenecks.
"""
from __future__ import annotations

import csv
import gc
import logging
import os
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from tracklab.pipeline.detectionlevel_module import DetectionLevelModule
from tracklab.utils.collate import default_collate, Unbatchable

log = logging.getLogger(__name__)

# ── timing acumulado ─────────────────────────────────────────────────────────
_T: dict = defaultdict(float)
_N: dict = defaultdict(int)

def _tock(key: str, t0: float):
    _T[key] += time.perf_counter() - t0
    _N[key] += 1

def _log_timings(out_dir: Path = None, print_summary: bool = True):
    import json
    rows = {}
    if print_summary:
        lines = ["", "[JNR-v2 TIMING] ══════════════════════════════════════════════"]
    for k in sorted(_T):
        total = _T[k]; n = _N[k]
        avg_ms = (total / n * 1000) if n else 0
        if print_summary:
            lines.append(f"  {k:<42s} total={total:7.2f}s  n={n:5d}  avg={avg_ms:7.1f}ms")
        rows[k] = {"total_s": round(total, 4), "n": n, "avg_ms": round(avg_ms, 2)}
    if print_summary:
        lines.append("[JNR-v2 TIMING] ══════════════════════════════════════════════\n")
        print("\n".join(lines), flush=True)

    # gravar em ficheiro JSON (sobrescreve — sempre a versão mais recente)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timing_file = out_dir / "jnr_v2_timing.json"
        with open(timing_file, "w") as f:
            json.dump(rows, f, indent=2)
        if print_summary:
            log.info(f"[JNR-v2] timing guardado em {timing_file}")

# ─── COCO skeleton pairs para desenho ─────────────────────────────────────────
_SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),        # braços
    (5,11),(6,12),(11,12),                  # tronco
    (11,13),(13,15),(12,14),(14,16),        # pernas
    (0,1),(0,2),(1,3),(2,4),               # face
]
_KP_NAMES = ['nose','l_eye','r_eye','l_ear','r_ear',
             'l_sh','r_sh','l_elb','r_elb','l_wr','r_wr',
             'l_hip','r_hip','l_kn','r_kn','l_ank','r_ank']

def _draw_pose_debug(img_bgr, kxy, kcf, roi_coords=None, rejected=False):
    """
    Devolve uma cópia do frame com:
      - skeleton desenhado
      - keypoints coloridos por confiança
      - bbox ROI a verde (ou vermelho se rejeitado)
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # skeleton
    for a, b in _SKELETON:
        if kcf[a] > 0.2 and kcf[b] > 0.2:
            pa = (int(kxy[a][0]), int(kxy[a][1]))
            pb = (int(kxy[b][0]), int(kxy[b][1]))
            cv2.line(out, pa, pb, (255, 200, 0), 2)

    # keypoints
    for i, (pt, cf) in enumerate(zip(kxy, kcf)):
        if cf > 0.1:
            color = (0, 255, 0) if cf > 0.5 else (0, 165, 255)
            cv2.circle(out, (int(pt[0]), int(pt[1])), 4, color, -1)
            cv2.putText(out, f"{_KP_NAMES[i]}:{cf:.2f}",
                        (int(pt[0])+4, int(pt[1])-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # ROI box
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        color = (0, 0, 255) if rejected else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = "REJECTED" if rejected else "CROP ROI"
        cv2.putText(out, label, (x1, max(y1-8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # face_conf / ears label (MAX — matches the actual rejection check)
    face_max = float(max(kcf[0], kcf[1], kcf[2]))
    ears_min = float(min(kcf[3], kcf[4]))
    sh_hip_min = float(min(kcf[5], kcf[6], kcf[11], kcf[12]))
    tw_val = abs(kxy[6][0] - kxy[5][0])
    cv2.putText(out, f"fmax={face_max:.2f} ears={ears_min:.2f}", (2, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    cv2.putText(out, f"sh_hip_min={sh_hip_min:.2f} tw={tw_val:.0f}", (2, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    return out


def _clamp(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w-1, int(x1)));  y1 = max(0, min(h-1, int(y1)))
    x2 = max(1, min(w,   int(x2)));  y2 = max(1, min(h,   int(y2)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _is_facing_camera(kcf) -> bool:
    """
    Retorna True se o jogador está claramente de frente (não mostra as costas).
    Só verifica nariz/olhos — orelhas NÃO são usadas (são visíveis mesmo de costas).
    Threshold conservador: 0.60 evita rejeitar jogadores a meio-giro.
    """
    return max(kcf[0], kcf[1], kcf[2]) > 0.60   # nariz ou olho claramente visível


def _extract_back_roi(img_rgb, kxy, kcf, pad_x, pad_y_top, roi_h_frac, w, h):
    """
    Dado keypoints YOLO já calculados, extrai crop RGB das costas.
    """
    ls, rs = kxy[5], kxy[6]
    lh, rh = kxy[11], kxy[12]
    sh_y = min(ls[1], rs[1]);  hi_y = max(lh[1], rh[1])
    cx   = (min(ls[0],rs[0]) + max(ls[0],rs[0])) / 2
    tw   = abs(rs[0]-ls[0]);   th   = hi_y - sh_y
    coords = _clamp(cx - tw/2*(1+pad_x), sh_y - th*pad_y_top,
                    cx + tw/2*(1+pad_x), sh_y + th*roi_h_frac, w, h)

    face_conf = float(max(kcf[0], kcf[1], kcf[2]))
    face_max  = max(kcf[0], kcf[1], kcf[2])
    ears_min  = min(kcf[3], kcf[4])
    sh_hip    = [round(float(kcf[i]), 2) for i in [5, 6, 11, 12]]

    reject_reason = None
    if _is_facing_camera(kcf):
        reject_reason = f"face_visible={face_max:.2f}"
    elif any(kcf[i] < 0.10 for i in [5, 6, 11, 12]):
        reject_reason = f"low_sh_hip={sh_hip}"
    elif tw < 10 or th < 10:
        reject_reason = f"small_torso tw={tw:.1f} th={th:.1f}"

    rejected = reject_reason is not None
    if rejected:
        log.debug(f"[JNR-v2] REJECT reason={reject_reason} crop={w}x{h}")
    if rejected or coords is None:
        return None, coords, rejected, face_conf
    x1,y1,x2,y2 = coords
    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None, coords, rejected, face_conf
    return crop, coords, rejected, face_conf


def _yolo_back_crop_from_pred(crop_rgb, pred, full_frame_bgr,
                               pad_x, pad_y_top, roi_h_frac,
                               debug_dir=None, frame_id=None):
    """
    FIX3: extrai ROI das costas usando resultado YOLO já calculado para o frame.
    crop_rgb  = crop do jogador (bbox do tracker) — usado para obter keypoints relativos
    pred      = resultado YOLO do frame completo
    full_frame_bgr = frame BGR (para debug)

    Estratégia: usar o crop_rgb directamente como imagem de entrada dos keypoints,
    porque já está restringido à bbox do jogador.
    """
    h, w = crop_rgb.shape[:2]
    img_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    # Reusa pred se disponível e tem keypoints, senão faz predict no crop
    if pred is not None and pred.keypoints is not None and len(pred.keypoints.xy) > 0:
        # encontrar a detecção YOLO mais próxima do centro do crop no frame completo
        # (heurística: usar a maior bbox)
        best_i, best_area = 0, 0
        if pred.boxes is not None and len(pred.boxes.xyxy) > 0:
            for i, box in enumerate(pred.boxes.xyxy):
                bx1,by1,bx2,by2 = box.cpu().numpy()
                a = (bx2-bx1)*(by2-by1)
                if a > best_area:
                    best_area, best_i = a, i
            kxy = pred.keypoints.xy[best_i].cpu().numpy()
            kcf = pred.keypoints.conf[best_i].cpu().numpy()
        else:
            return None, None
    else:
        # fallback: fazer predict no crop individual
        try:
            p = _POSE_MODEL_REF.predict(img_bgr, conf=0.25, verbose=False, workers=0)[0]
        except Exception:
            return None, None
        if p.keypoints is None or len(p.keypoints.xy) == 0:
            return None, None
        kxy = p.keypoints.xy[0].cpu().numpy()
        kcf = p.keypoints.conf[0].cpu().numpy()

    crop_result, coords, rejected, face_conf = _extract_back_roi(
        crop_rgb, kxy, kcf, pad_x, pad_y_top, roi_h_frac, w, h)

    if debug_dir and frame_id is not None:
        t0 = time.perf_counter()
        suffix = "REJECTED" if rejected else "full_pose"
        debug_img = _draw_pose_debug(img_bgr, kxy, kcf, roi_coords=coords, rejected=rejected)
        cv2.imwrite(str(debug_dir / f"frame_{frame_id}_{suffix}.jpg"), debug_img)
        _tock("debug_imwrite_pose", t0)
        if crop_result is not None:
            t0 = time.perf_counter()
            cv2.imwrite(str(debug_dir / f"frame_{frame_id}_crop.jpg"),
                        cv2.cvtColor(crop_result, cv2.COLOR_RGB2BGR))
            _tock("debug_imwrite_crop", t0)

    return crop_result, coords


# referência global ao pose model para fallback (preenchida em __init__)
_POSE_MODEL_REF = None


def _yolo_back_crop(img_rgb, pose_model, pose_conf,
                    pad_x, pad_y_top, roi_h_frac,
                    debug_dir=None, frame_id=None):
    """
    YOLO pose → crop RGB das costas.
    Recebe img_rgb (formato tracklab: RGB).
    Devolve (crop_rgb, roi_coords) ou (None, None).
    Se debug_dir fornecido grava frame com pose + ROI (convertido para BGR para cv2.imwrite).
    """
    h, w = img_rgb.shape[:2]
    # YOLO espera BGR
    t0 = time.perf_counter()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _tock("yolo_bgr_convert", t0)

    try:
        t0 = time.perf_counter()
        pred = pose_model.predict(img_bgr, conf=pose_conf, verbose=False, workers=0)[0]
        _tock("yolo_predict", t0)
    except Exception as e:
        log.debug(f"[YOLO] {e}")
        return None, None

    if pred.keypoints is None or len(pred.keypoints.xy) == 0:
        if debug_dir and frame_id is not None:
            cv2.imwrite(str(debug_dir / f"frame_{frame_id}_NO_POSE.jpg"), img_bgr)
        return None, None

    # maior pessoa
    best_i, best_area = 0, 0
    if pred.boxes is not None:
        for i, box in enumerate(pred.boxes.xyxy):
            bx1,by1,bx2,by2 = box.cpu().numpy()
            a = (bx2-bx1)*(by2-by1)
            if a > best_area:
                best_area, best_i = a, i

    kxy = pred.keypoints.xy[best_i].cpu().numpy()
    kcf = pred.keypoints.conf[best_i].cpu().numpy()

    face_conf = float(max(kcf[0], kcf[1], kcf[2]))

    # calcular ROI sempre (para debug mesmo se rejeitado)
    ls, rs = kxy[5], kxy[6]
    lh, rh = kxy[11], kxy[12]
    sh_y = min(ls[1], rs[1]);  hi_y = max(lh[1], rh[1])
    cx   = (min(ls[0],rs[0]) + max(ls[0],rs[0])) / 2
    tw   = abs(rs[0]-ls[0]);   th   = hi_y - sh_y
    coords = _clamp(cx - tw/2*(1+pad_x), sh_y - th*pad_y_top,
                    cx + tw/2*(1+pad_x), sh_y + th*roi_h_frac, w, h)

    rejected = (
        _is_facing_camera(kcf)
        or any(kcf[i] < 0.10 for i in [5,6,11,12])
        or tw < 10 or th < 10
    )

    # debug: frame com pose — cv2.imwrite espera BGR
    if debug_dir and frame_id is not None:
        debug_img = _draw_pose_debug(img_bgr, kxy, kcf,
                                     roi_coords=coords, rejected=rejected)
        suffix = "REJECTED" if rejected else "full_pose"
        cv2.imwrite(str(debug_dir / f"frame_{frame_id}_{suffix}.jpg"), debug_img)

    if rejected or coords is None:
        return None, None

    x1,y1,x2,y2 = coords
    crop_rgb = img_rgb[y1:y2, x1:x2]
    if crop_rgb.size == 0:
        return None, None

    # debug: gravar crop — cv2.imwrite espera BGR
    if debug_dir and frame_id is not None:
        t0 = time.perf_counter()
        cv2.imwrite(str(debug_dir / f"frame_{frame_id}_crop.jpg"),
                    cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        _tock("debug_imwrite_crop", t0)

    return crop_rgb, coords


class TrackletsJNRv2(DetectionLevelModule):
    input_columns  = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, batch_size, device,
                 pose_weights="/home/joao/soccernet/weights/yolo26m-pose.pt",
                 pose_conf=0.25,
                 pad_x=0.55, pad_y_top=0.10, roi_height_frac=0.90,
                 qwen_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
                 min_crops_before_qwen=3,
                 min_valid_windows=2,
                 window_min_agreement=0.75,
                 track_min_agreement=0.75,
                 finalize_min_crops=3,
                 # Filtro de qualidade do crop antes de ir ao Qwen
                 min_crop_h=30, min_crop_w=25, min_sharpness=40.0,
                 save_debug_crops=True,
                 # Batch multi-tracklet: agrupa N tracklets por chamada Qwen
                 max_tracklets_per_qwen_batch=4,
                 max_total_crops_per_batch=20,
                 # Distilação: grava crops aceites + labels para treinar modelo rápido
                 save_distill_crops=False,
                 tracking_dataset=None, **kwargs):
        super().__init__(batch_size=batch_size)
        self.device                      = device
        self.pose_conf                   = pose_conf
        self.pad_x                       = pad_x
        self.pad_y_top                   = pad_y_top
        self.roi_height_frac             = roi_height_frac
        self.qwen_model_path             = qwen_model_path
        self.min_crops_before_qwen       = min_crops_before_qwen
        self.min_valid_windows           = min_valid_windows
        self.window_min_agreement        = window_min_agreement
        self.track_min_agreement         = track_min_agreement
        self.finalize_min_crops          = finalize_min_crops
        self._min_crop_h                 = min_crop_h
        self._min_crop_w                 = min_crop_w
        self._min_sharpness              = min_sharpness
        self.save_debug_crops            = save_debug_crops
        self.max_tracklets_per_qwen_batch = max_tracklets_per_qwen_batch
        self.max_total_crops_per_batch   = max_total_crops_per_batch
        self.save_distill_crops          = save_distill_crops

        # tid → [(fid, crop_rgb)]
        self.pending_crops: Dict[Any, List[Tuple]] = defaultdict(list)
        # tid → [str|None, ...]
        self.qwen_results: Dict[Any, List] = defaultdict(list)
        # tid → [(sharpness, crop_rgb)] top-5 sharpest crops, for rescue pass
        self.best_crops: Dict[Any, List[Tuple[float, np.ndarray]]] = defaultdict(list)
        self._best_crops_k = 5

        self._debug_dir    = None   # root: jnr_v2_debug/
        self._vid_dir      = None   # por vídeo: jnr_v2_debug/SNGS-XXX/
        self._current_vid  = None   # e.g. "021"
        self._distill_dir  = None
        self._distill_csv  = None   # csv.writer aberto em finalize
        self._pose_on_gpu  = True

        log.info(f"[JNR-v2] Carregando YOLO: {pose_weights}")
        from ultralytics import YOLO
        self.pose_model = YOLO(pose_weights)
        global _POSE_MODEL_REF
        _POSE_MODEL_REF = self.pose_model
        log.info("[JNR-v2] YOLO pronto.")

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        """Crop do bbox do jogador — idêntico à baseline."""
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True)
        crop = image[t:b, l:r]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((10, 10, 3), dtype=np.uint8)
        return {"img": Unbatchable([crop])}

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        t_process = time.perf_counter()

        # ── setup root debug dir (apenas na primeira chamada) ───────────────
        if self._debug_dir is None:
            run_path = Path(os.getenv("HYDRA_RUN_DIR", os.getcwd()))
            self._debug_dir = run_path / "jnr_v2_debug"
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            if self.save_distill_crops:
                self._distill_dir = run_path / "distill_crops"
                self._distill_dir.mkdir(parents=True, exist_ok=True)
                self._open_distill_csv(run_path)

        # ── detectar vídeo actual e criar subdir por vídeo ───────────────────
        if not detections.empty and "image_id" in detections.columns:
            fid_sample = str(detections["image_id"].iloc[0])
            # image_id ex: "2021000001" → seq "021"
            vid = fid_sample[1:4] if len(fid_sample) >= 4 else fid_sample
            if vid != self._current_vid:
                self._current_vid = vid
                self._vid_dir = self._debug_dir / f"SNGS-{vid}"
                self._vid_dir.mkdir(parents=True, exist_ok=True)
                log.info(f"[JNR-v2] vídeo SNGS-{vid} → {self._vid_dir}")

        detections["jersey_number_detection"]  = None
        detections["jersey_number_confidence"] = 0.0

        t0 = time.perf_counter()
        images_np = [img.cpu().numpy() if torch.is_tensor(img) else img
                     for img in batch["img"]]
        _tock("process_to_numpy", t0)

        # ── 1. YOLO batch por jogador: 1 predict por crop (correcto) ─────────
        # Cada deteção tem o seu próprio crop → YOLO processa todos em batch GPU.
        # O FIX3 anterior era bugado: usava o YOLO do jogador 0 para todos os
        # jogadores do mesmo frame, produzindo keypoints errados para jogadores 1..N.
        t0 = time.perf_counter()

        tdir_base = (self._vid_dir or self._debug_dir) if self.save_debug_crops else None
        yolo_bgrs: List[np.ndarray] = []
        crop_meta: List[tuple]      = []   # (fid, tid, crop_rgb, img_bgr, tdir)

        for i, (idx, det) in enumerate(detections.iterrows()):
            if i >= len(images_np) or images_np[i] is None:
                continue
            tid = det.get("track_id")
            if tid is None:
                continue
            crop_img = images_np[i]
            if crop_img is None or crop_img.size == 0:
                continue
            fid    = det.get("image_id", idx)
            img_bgr = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            tdir   = (tdir_base / f"tracklet_{tid}") if tdir_base else None
            if tdir:
                tdir.mkdir(exist_ok=True)
            yolo_bgrs.append(img_bgr)
            crop_meta.append((fid, tid, crop_img, img_bgr, tdir))

        _tock("yolo_bgr_convert", t0)

        n_accepted = 0
        if yolo_bgrs:
            try:
                t0 = time.perf_counter()
                preds = self.pose_model.predict(
                    yolo_bgrs, conf=self.pose_conf, verbose=False, workers=0
                )
                _tock("yolo_predict", t0)
            except Exception as e:
                log.debug(f"[YOLO] {e}")
                preds = [None] * len(yolo_bgrs)

            t0 = time.perf_counter()
            for pred, (fid, tid, crop_img, img_bgr, tdir) in zip(preds, crop_meta):
                crop_rgb, _ = _yolo_back_crop_from_pred(
                    crop_img, pred, img_bgr,
                    self.pad_x, self.pad_y_top, self.roi_height_frac,
                    debug_dir=tdir, frame_id=fid,
                )
                if crop_rgb is None:
                    continue
                # ── Filtro de qualidade: rejeitar crops demasiado pequenos/desfocados ──
                ch, cw = crop_rgb.shape[:2]
                if ch < self._min_crop_h or cw < self._min_crop_w:
                    continue
                gray      = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness < self._min_sharpness:
                    continue
                self.pending_crops[tid].append((fid, crop_rgb))
                n_accepted += 1
                # keep top-K sharpest crops per tracklet for rescue pass
                self.best_crops[tid].append((sharpness, crop_rgb))
                if len(self.best_crops[tid]) > self._best_crops_k:
                    self.best_crops[tid].sort(key=lambda x: x[0], reverse=True)
                    self.best_crops[tid] = self.best_crops[tid][:self._best_crops_k]
            _tock("process_yolo_loop", t0)

        # ── 2. Qwen: disparar janelas prontas em batch ────────────────────────
        ready_tids = [
            tid for tid, crops in self.pending_crops.items()
            if len(crops) >= self.min_crops_before_qwen
        ]

        if ready_tids:
            log.info(f"[JNR-v2] batch: {len(detections)} dets | accepted={n_accepted} | "
                     f"ready_tids={len(ready_tids)} → Qwen")
            # Sync + free YOLO cache before Qwen to avoid CUDA state corruption.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            t0 = time.perf_counter()
            # Primeira janela de cada tracklet → solo (evita batch contamination)
            # Janelas repetidas → batch normal de max_tracklets_per_qwen_batch
            first_time = [t for t in ready_tids if not self.qwen_results[t]]
            repeat     = [t for t in ready_tids if self.qwen_results[t]]
            if first_time:
                log.info(f"[JNR-v2] {len(first_time)} first-window solo + {len(repeat)} repeat batch")
            for tid in first_time:
                self._run_qwen_solo(tid)
            for i in range(0, len(repeat), self.max_tracklets_per_qwen_batch):
                sub = repeat[i: i + self.max_tracklets_per_qwen_batch]
                self._run_qwen_batch(sub)
            _tock("process_qwen_windows", t0)
        else:
            log.debug(f"[JNR-v2] batch: {len(detections)} dets | accepted={n_accepted} | "
                      f"nenhum tracklet pronto para Qwen")

        # ── 3. Propagar melhor resultado para o batch atual ───────────────────
        t0 = time.perf_counter()
        self._propagate_all(detections)
        _tock("process_propagate", t0)

        _tock("process_total", t_process)
        _log_timings(out_dir=(self._vid_dir or self._debug_dir), print_summary=False)
        return detections



    def _record_result(self, tid: Any, nr: Optional[str], engine: str,
                       fid_crops: List[Tuple], crops_bgr: List[np.ndarray],
                       incomplete: bool):
        """Regista resultado, imprime log, grava debug e distilação."""
        self.qwen_results[tid].append(nr)
        flag = " [incomplete]" if incomplete else ""
        print(
            f"[JNR-v2] tracklet={tid}{flag} [{engine}] | {len(crops_bgr)} crops"
            f" → nr={nr!r} | historial={self.qwen_results[tid]}",
            flush=True,
        )
        if self.save_debug_crops and (self._vid_dir or self._debug_dir):
            t0 = time.perf_counter()
            tdir = (self._vid_dir or self._debug_dir) / f"tracklet_{tid}"
            tdir.mkdir(exist_ok=True)
            tag = nr if nr else "None"
            for (fid, _), crop_bgr in zip(fid_crops, crops_bgr):
                cv2.imwrite(str(tdir / f"frame_{fid}_{engine.lower()}{tag}.jpg"), crop_bgr)
            _tock("debug_imwrite", t0)
        if self.save_distill_crops and nr is not None and self._distill_dir:
            t0 = time.perf_counter()
            self._log_distill(tid, fid_crops, crops_bgr, nr)
            _tock("distill_log", t0)

    def _run_qwen_batch(self, tids: List, incomplete: bool = False):
        """
        Chama Qwen para uma lista de tracklets em batch.
        Usado para janelas repetidas (tracklets que já tiveram pelo menos 1 chamada Qwen).
        """
        from sn_gamestate.jersey.tracklets_jnr_v2 import qwen_ocr_multi_tracklet

        batch_fid_crops: Dict[Any, List[Tuple]] = {
            tid: self.pending_crops.pop(tid, []) for tid in tids
        }
        t0 = time.perf_counter()
        batch_bgr: Dict[Any, List[np.ndarray]] = {
            tid: [cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for _, c in fid_crops]
            for tid, fid_crops in batch_fid_crops.items()
            if fid_crops
        }
        _tock("bgr_convert", t0)

        if not batch_bgr:
            return

        n_crops = sum(len(v) for v in batch_bgr.values())
        t0 = time.perf_counter()
        results = qwen_ocr_multi_tracklet(
            batch_bgr, self.qwen_model_path,
            max_total_crops=self.max_total_crops_per_batch,
        )
        t_qwen = time.perf_counter() - t0
        _tock("qwen_inference", t0)
        log.info(f"[JNR-v2] Qwen batch: {len(batch_bgr)} tracklets | {n_crops} crops | {t_qwen:.2f}s")

        for tid in batch_bgr:
            result = results.get(tid)
            nr = result[0] if result else None
            self._record_result(tid, nr, "QWEN",
                                batch_fid_crops[tid], batch_bgr.get(tid, []), incomplete)

    def _run_qwen_solo(self, tid: Any, incomplete: bool = False):
        """
        Chamada Qwen solo para a PRIMEIRA janela de um tracklet.
        Evita batch contamination: quando muitos tracklets disparam simultaneamente
        na primeira janela, misturar crops de qualidade variável faz o Qwen devolver
        None para todos — incluindo os com número claramente visível.
        """
        from sn_gamestate.jersey.tracklets_jnr_v2 import qwen_ocr_multi_tracklet

        fid_crops = self.pending_crops.pop(tid, [])
        if not fid_crops:
            return

        t0 = time.perf_counter()
        crops_bgr = [cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for _, c in fid_crops]
        _tock("bgr_convert", t0)

        t0 = time.perf_counter()
        results = qwen_ocr_multi_tracklet(
            {tid: crops_bgr}, self.qwen_model_path,
            max_total_crops=min(3, len(crops_bgr)),
        )
        t_qwen = time.perf_counter() - t0
        _tock("qwen_inference", t0)

        result = results.get(tid)
        nr = result[0] if result else None
        print(f"[JNR-v2] solo first-window tid={tid} → {nr!r} ({t_qwen:.2f}s)", flush=True)
        self._record_result(tid, nr, "QWEN", fid_crops, crops_bgr, incomplete)

    def _log_distill(self, tid, fid_crops: List[Tuple],
                     crops_bgr: List[np.ndarray], jersey_number: str):
        """
        Grava cada crop aceite em disco e regista no CSV de distilação.
        Formato: distill_crops/<jersey_number>/tracklet_<tid>_frame_<fid>.jpg
        CSV: crop_path, tid, frame_id, jersey_number
        """
        if self._distill_dir is None:
            return
        label_dir = self._distill_dir / jersey_number
        label_dir.mkdir(parents=True, exist_ok=True)

        for (fid, _), crop_bgr in zip(fid_crops, crops_bgr):
            fname = f"tracklet_{tid}_frame_{fid}.jpg"
            fpath = label_dir / fname
            cv2.imwrite(str(fpath), crop_bgr)

            # CSV aberto em append mode via _open_distill_csv()
            if self._distill_csv is not None:
                self._distill_csv.writerow([
                    str(fpath), str(tid), str(fid), jersey_number
                ])

    def _open_distill_csv(self, run_path: Path):
        """Abre (ou cria) o CSV de distilação e escreve o header se necessário."""
        csv_path = run_path / "distill_labels.csv"
        write_header = not csv_path.exists()
        self._distill_csv_file = open(csv_path, "a", newline="")
        self._distill_csv = csv.writer(self._distill_csv_file)
        if write_header:
            self._distill_csv.writerow(["crop_path", "tid", "frame_id", "jersey_number"])
        log.info(f"[JNR-v2] Distill CSV: {csv_path}")

    def finalize_ocr(self, detections: pd.DataFrame) -> pd.DataFrame:
        """
        Chamado no fim do pipeline.
        1. Processa tracklets com crops pendentes que nunca completaram uma janela.
        2. Aplica nível 2: track_min_agreement sobre qwen_results[tid].
        3. Propaga nr final para todo o DataFrame de deteções.
        4. Imprime sumário de timing.
        """
        t_fin = time.perf_counter()

        # tracklets com crops sobrantes (janela incompleta)
        # FIX-D: só chamar Qwen se tiver >= finalize_min_crops crops
        incomplete = [
            tid for tid, crops in self.pending_crops.items()
            if len(crops) >= self.finalize_min_crops
        ]
        skipped = [
            tid for tid, crops in self.pending_crops.items()
            if 0 < len(crops) < self.finalize_min_crops
        ]
        if skipped:
            log.info(f"[JNR-v2] finalize: ignorados {len(skipped)} tracklets com "
                     f"<{self.finalize_min_crops} crops (FIX-D): {skipped}")
        if incomplete:
            log.info(f"[JNR-v2] finalize: {len(incomplete)} tracklets incompletos → batch Qwen")
            for i in range(0, len(incomplete), self.max_tracklets_per_qwen_batch):
                sub = incomplete[i: i + self.max_tracklets_per_qwen_batch]
                self._run_qwen_batch(sub, incomplete=True)

        # Fechar CSV de distilação
        if hasattr(self, "_distill_csv_file") and self._distill_csv_file:
            self._distill_csv_file.close()
            self._distill_csv_file = None
            self._distill_csv = None

        # nível 2: track_min_agreement
        results_summary = []
        for tid in sorted(self.qwen_results.keys(), key=lambda x: float(x)):
            jn, conf = self._best_for_track(tid)
            if jn:
                line = f"[JNR-v2] finalize tracklet={tid} → {jn!r} conf={conf:.0%} historial={self.qwen_results[tid]}"
            else:
                line = f"[JNR-v2] finalize tracklet={tid} → sem consenso historial={self.qwen_results[tid]}"
            print(line, flush=True)
            results_summary.append({"tid": tid, "jersey": jn, "conf": conf,
                                     "historial": self.qwen_results[tid]})

        # ── Rescue pass: solo Qwen com o crop mais nítido para tracklets sem consenso ─
        unresolved = [
            tid for tid in sorted(self.qwen_results.keys(), key=lambda x: float(x))
            if self._best_for_track(tid)[0] is None
            and tid in self.best_crops
            and self.best_crops[tid]
        ]
        if unresolved:
            from sn_gamestate.jersey.tracklets_jnr_v2 import qwen_ocr_multi_tracklet
            log.info(f"[JNR-v2] rescue pass: {len(unresolved)} tracklets sem consenso")
            rescued = 0
            for tid in unresolved:
                # top-3 sharpest crops, solo call (no batch contamination)
                best = sorted(self.best_crops[tid], key=lambda x: x[0], reverse=True)[:3]
                crops_bgr = [cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for _, c in best]
                result = qwen_ocr_multi_tracklet(
                    {tid: crops_bgr}, self.qwen_model_path, max_total_crops=3
                )
                if result.get(tid):
                    jn_rescue, conf_rescue = result[tid]
                    self.qwen_results[tid].append(jn_rescue)
                    print(f"[JNR-v2] rescue tid={tid} → {jn_rescue!r} (sharpness-top3)", flush=True)
                    # update summary entry
                    for entry in results_summary:
                        if entry["tid"] == tid:
                            entry["jersey"] = jn_rescue
                            entry["conf"]   = round(conf_rescue, 3)
                            entry["historial"].append(jn_rescue)
                            rescued += 1
                            break
                else:
                    print(f"[JNR-v2] rescue tid={tid} → None", flush=True)
            log.info(f"[JNR-v2] rescue pass: {rescued}/{len(unresolved)} recuperados")

        # ── determinar dir deste vídeo (pode não ter batches ainda em edge cases) ──
        out_dir = self._vid_dir or self._debug_dir

        # ── Gravar resultados em JSON ─────────────────────────────────────
        if out_dir is not None:
            import json as _json
            results_file = out_dir / "jnr_results.json"
            with open(results_file, "w") as f:
                _json.dump(results_summary, f, indent=2, default=str)
            log.info(f"[JNR-v2] resultados → {results_file}")

        self._propagate_all(detections)
        _tock("finalize_total", t_fin)

        # ── sumário de timing por vídeo — print + JSON ───────────────────
        _log_timings(out_dir=out_dir)

        # ── Reset estado para próximo vídeo ──────────────────────────────
        self.pending_crops.clear()
        self.qwen_results.clear()
        self.best_crops.clear()
        _T.clear()
        _N.clear()
        self._vid_dir     = None
        self._current_vid = None

        # Liberta Qwen da VRAM após cada vídeo — evita OOM quando calibração/pitch
        # correm no vídeo seguinte. O modelo é recarregado do disco (~5s) no início
        # do próximo vídeo. Apagar as referências globais é seguro com bitsandbytes.
        from sn_gamestate.jersey.tracklets_jnr_v2 import unload_qwen_from_vram
        unload_qwen_from_vram()

        return detections

    def _best_for_track(self, tid):
        """
        Nível 2: nr mais frequente nas janelas de qwen_results[tid].
        Critérios de aceitação:
          1. Mínimo min_valid_windows janelas não-None
          2. O melhor número aparece em >= track_min_agreement de TODAS as janelas
             (None conta CONTRA o número)
          3. Se falhar o threshold global, tenta segmentação temporal — o tracker
             pode ter fundido dois jogadores numa única tracklet.
        """
        results = self.qwen_results.get(tid, [])
        valid   = [r for r in results if r is not None]
        if len(valid) < self.min_valid_windows:
            return None, 0.0
        counter = Counter(valid)
        best    = counter.most_common(1)[0][0]
        conf    = counter[best] / len(results)
        if conf >= self.track_min_agreement:
            return best, round(conf, 3)
        # Fallback: tentar segmentação temporal
        return self._best_temporal_segment(results)

    def _best_temporal_segment(self, results):
        """
        Procura um ponto de corte onde um dos segmentos satisfaz o threshold.
        Útil quando o tracker fundiu dois jogadores (ID switch).
        Só tenta em historiais com ≥ 10 janelas para evitar overfit em sequências curtas.
        Retorna o número com maior conf no melhor segmento encontrado.
        """
        n = len(results)
        if n < 10:
            return None, 0.0
        best_conf = 0.0
        best_num  = None
        # testar splits entre 20% e 80% do historial
        for split in range(max(1, n // 5), min(n, 4 * n // 5)):
            for seg in (results[:split], results[split:]):
                if len(seg) < 8:
                    continue
                valid_seg = [r for r in seg if r is not None]
                if len(valid_seg) < self.min_valid_windows:
                    continue
                cnt = Counter(valid_seg)
                top_num, top_n = cnt.most_common(1)[0]
                seg_conf = top_n / len(seg)
                if seg_conf >= self.track_min_agreement and seg_conf > best_conf:
                    best_conf = seg_conf
                    best_num  = top_num
        return best_num, round(best_conf, 3)

    def _propagate_all(self, detections):
        for tid in detections["track_id"].dropna().unique():
            jn, conf = self._best_for_track(tid)
            if jn is None:
                continue
            mask = detections["track_id"] == tid
            detections.loc[mask, "jersey_number_detection"]  = jn
            detections.loc[mask, "jersey_number_confidence"] = conf
