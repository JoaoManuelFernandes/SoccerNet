# tracklets_ocr.py
"""
Pipeline principal de OCR para tracklets
"""

import pandas as pd
import torch
import numpy as np
import os
import math
import logging
import cv2
import time
import datetime

import re
from collections import defaultdict, Counter
import gc, paddle, torch
print("paddle a usar:" + paddle.device.get_device())        # deve devolver 'gpu:0'

from paddleocr import PaddleOCR
#from mmocr.apis import TextDetInferencer, TextRecInferencer
#from mmocr.utils import bbox2poly, crop_img, poly2bbox

from mmengine.registry import init_default_scope
init_default_scope('mmpose')

#from mmpose.apis import MMPoseInferencer

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)

from sn_gamestate.jersey.tracklets_utils import (
    run_hrnet_pose_inference,
    is_facing_away,
    crop_back_region,
    debug_save_image,
    save_images_by_tracklet,
    COCO_SKELETON,
    compute_pose_metrics,
    save_pose_debug_data,is_facing_away_trunk_motion
)


log = logging.getLogger(__name__)

class TrackletsOCR(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(
        self,
        batch_size,
        device,
        model_dir=None,
        ocr_decision_strategy="confidence_sum",
        min_votes = 2,
        confidence_threshold = 0.85,
        jnr_batch_frames = 5,
        use_superres = True,
        debug_paddleocr = False,
        ocr_only_model = None, 
        pose_model_name = "td-hm_hrnet-w32_8xb64-210e_coco-256x192",
        angle_threshold_1 = 175,
        angle_threshold_2 = 150,
        angle_threshold_3 = 145,
        score_threshold_1 = 0.7,
        score_threshold_2 = 0.8,
        angle_trunk_motion_threshold = 120,
        min_percent_away = 70.0,
        **kwargs
    ):

        super().__init__(batch_size=batch_size)
        self.device = device
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.ocr_decision_strategy = ocr_decision_strategy
        self.JNRDebug =False
        self.imageDebug = False
        self.confidence_threshold = confidence_threshold
        self.jnr_batch_frames = jnr_batch_frames
        self.use_superres = use_superres
        self.pose_model_name = pose_model_name

        self.pose_debug_data = defaultdict(list)

        self.best_by_track = {}

        self.tracklet_images_global = {}
        self.comparison_rows = []
        self.debug_paddleocr = debug_paddleocr
        self.tracklet_debug_data = defaultdict(list) 

        # Buffer de detecções por bloco de frames, lógica de decisão/votação por bloco em should_replace, e logging detalhado das detecções por modelo/track para exportação ao final do pipeline.
        self.tracklet_detection_buffer = defaultdict(list)  # Para logging e decisão por bloco
        self.tracklet_detection_logs = defaultdict(list)  # Para logging detalhado por track/modelo
        self.confidence_sum_buffer = defaultdict(lambda: defaultdict(float))  # Para confidence_sum
        self.last_majority = defaultdict(lambda: None)  # Para sliding_persistence
        self.persistence_count = defaultdict(lambda: 0)  # Para sliding_persistence

        # MMOCR baseline
        #self.textdetinferencer = TextDetInferencer('dbnet_resnet18_fpnc_1200e_icdar2015', device=device)
        #self.textrecinferencer = TextRecInferencer('SAR', device=device)

        # --- Apenas HRNet/MMPose ---
        from mmpose.apis import MMPoseInferencer
        valid_models = MMPoseInferencer.list_models(scope='mmpose')
        if self.pose_model_name not in valid_models:
            raise ValueError(f"Modelo de pose '{self.pose_model_name}' não suportado. Escolha um dos modelos válidos: {valid_models}")
        self.pose_model = MMPoseInferencer(self.pose_model_name)
        self.pose_infer_fn = run_hrnet_pose_inference

        # --- Garantir que os modelos PaddleOCR v4 estão presentes ---
        # --- Usar diretório self.model_dir do config para modelos ---
        det_model_dir = os.path.join(self.model_dir, 'paddleocr', 'ch_PP-OCRv4_det_infer')
        rec_model_dir = os.path.join(self.model_dir, 'paddleocr', 'ch_PP-OCRv4_rec_infer')
        # URLs oficiais dos modelos (ajuste se necessário)
        det_url = 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar'
        rec_url = 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar'
        # Função para baixar e extrair se necessário
        def download_and_extract(model_dir, url):
            if not os.path.exists(model_dir):
                import requests, tarfile
                os.makedirs(model_dir, exist_ok=True)
                tar_path = model_dir + '.tar'
                print(f"Baixando modelo {url} ...")
                r = requests.get(url, allow_redirects=True)
                with open(tar_path, 'wb') as f:
                    f.write(r.content)
                print(f"Extraindo {tar_path} ...")
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=os.path.dirname(model_dir))
                os.remove(tar_path)
        download_and_extract(det_model_dir, det_url)
        download_and_extract(rec_model_dir, rec_url)
        # Inicialização dos modelos OCR
        self.ocr_models = {}
        # Permite escolher rodar apenas um modelo OCR para máxima velocidade
        self.ocr_only_model = ocr_only_model

        # PaddleOCR v4 com float16 se suportado
        self.ocr_models['ppocr_v4'] = PaddleOCR(
            use_gpu=True,
            det=True,
            cls=False,
            rec=True,
            det_algorithm='DB',
            rec_algorithm='SVTR_LCNet',
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            det_limit_side_len=960,
            det_db_thresh=0.1,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0,
            drop_score=0.2,
            det_limit_type='max',
            use_dilation=True,
            precision='fp16'  # otimização para GPU moderna
        )

        self.paddle_ocr = PaddleOCR(
            det=True,
            cls=False,
            rec=True,
            use_gpu=True,
            use_tensorrt=False,
            precision='fp32',
            det_algorithm='DB',
            rec_algorithm='SVTR_LCNet',
            det_db_thresh=0.1,           # Reduzido para captar contornos mais sutis
            det_db_box_thresh=0.3,       # Reduzido para aceitar detecções menos confiantes
            det_db_unclip_ratio=2.0,     # Aumentado para expandir a região de detecção
            drop_score=0.2,              # Reduzido para aceitar reconhecimentos menos confiantes
            det_limit_side_len=2240,     # Aumentado para melhor resolução
            det_limit_type='max',
            max_batch_size=10,
            use_dilation=True            # Adiciona dilatação para melhorar detecção
        )


        # 2. PP-OCRv3 (mantido para comparação)
        self.ocr_models['ppocr_v3'] = self.paddle_ocr

        try:
            # 3. EasyOCR
            import easyocr
            self.ocr_models['easyocr'] = easyocr.Reader(['en'], gpu=True)
        except ImportError:
            if self.JNRDebug:
                print("EasyOCR não instalado")

        # Super-resolução: RRDBNet + RealESRGANer
        self.rrdb = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        weight_path = "/home/joao/soccernet/pretrained_models/realesrgan/RealESRGAN_x4plus.pth"
        self.upscaler = RealESRGANer(
            scale=4,
            model_path=weight_path,
            model=self.rrdb,
            tile=0,
            pre_pad=0,
            half=True
        )
        self.min_votes = min_votes# mínimo de vezes seguidas que o número deve aparecer para ser aceito
        import datetime
        self.timing_log_path = os.path.join(os.getenv("HYDRA_RUN_DIR", os.getcwd()), "timing_log.txt")
        with open(self.timing_log_path, "a") as f:
            f.write(f"\n==== NOVA EXECUÇÃO: {datetime.datetime.now()} ====" + "\n")
        self.min_percent_away = min_percent_away
        self.angle_threshold_1 = angle_threshold_1
        self.angle_threshold_2 = angle_threshold_2
        self.angle_threshold_3 = angle_threshold_3
        self.score_threshold_1 = score_threshold_1
        self.score_threshold_2 = score_threshold_2
        self.angle_trunk_motion_threshold =angle_trunk_motion_threshold


        # Para análise: buffer para gravar orientação do tronco, direção do movimento, ângulo entre, etc.
        self.trunk_motion_debug = defaultdict(list)
    def no_jersey_number(self):
        return None, 0

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        """
        Faz o pré-processamento: recorta da imagem original (BGR ou RGB) o bounding box do jogador.
        Em seguida, empacota num dicionário 'batch' para processar no método 'process'.
        """
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]

        # Evita problemas de crops vazios
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((10, 10, 3), dtype=np.uint8)

        # Passa a lista de 1 imagem dentro de Unbatchable
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }

        return batch

    def extract_numbers(self, text):
        """
        Retorna somente os dígitos contidos na string 'text'.
        """
        number = ''
        for char in text:
            if char.isdigit():
                number += char
        return number if number != '' else None

    def choose_best_jersey_number(self, jersey_numbers, jn_confidences):
        """
        Dado uma lista de 'jersey_numbers' e suas 'jn_confidences',
        retorna (numero_com_maior_confiança, valor_da_confiança).
        """
        if len(jersey_numbers) == 0:
            return self.no_jersey_number()
        else:
            jn_confidences = np.array(jn_confidences)
            idx_sort = np.argsort(jn_confidences)
            # pega o de maior confiança
            return jersey_numbers[idx_sort[-1]], jn_confidences[idx_sort[-1]]

    def extract_jersey_numbers_from_ocr(self, prediction):
        """
        Lê o resultado de uma inferência (prediction) e tenta extrair
        os dígitos como número de camisa.
        """
        jersey_numbers = []
        jn_confidences = []
        if 'rec_texts' not in prediction or 'rec_scores' not in prediction:
            return self.no_jersey_number()

        for txt, conf in zip(prediction['rec_texts'], prediction['rec_scores']):
            jn = self.extract_numbers(txt)
            if jn is not None:
                jersey_numbers.append(jn)
                jn_confidences.append(conf)

        jersey_number, jn_confidence = self.choose_best_jersey_number(
            jersey_numbers,
            jn_confidences
        )
        # Exemplo simples: assume que só precisamos de até 2 dígitos
        if jersey_number is not None:
            jersey_number = jersey_number[:2]
        return jersey_number, jn_confidence

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):

        run_path = os.getenv("HYDRA_RUN_DIR", os.getcwd())
        log.info(f"🔍 Diretório Hydra: {run_path}")

        t0_total = time.perf_counter()
        timing_rows = []
        def log_timing(etapa, track_id=None, extra=None, t_ini=None, t_fim=None):
            t_now = time.perf_counter()
            t_total = t_now - t0_total
            t_delta = (t_fim-t_ini) if (t_ini is not None and t_fim is not None) else None
            dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = {
                'etapa': etapa,
                'track_id': track_id,
                'tempo': t_delta,
                'tempo_total': t_total,
                'extra': extra,
                'datetime': dt
            }
            timing_rows.append(row)
            with open(self.timing_log_path, "a") as f:
                f.write(f"{etapa},{track_id},{t_delta},{t_total},{extra},{dt}\n")

        # 1) Preparar colunas iniciais
        t_ini = time.perf_counter()
        images_np = [img.cpu().numpy() for img in batch['img']]
        del batch['img']
        detections['jersey_number_detection'] = None
        detections['jersey_number_confidence'] = 0.0
        detections['ocr_tracklet_pose_jn']   = np.nan
        detections['ocr_tracklet_pose_conf'] = 0.0
        log_timing('preparar_colunas', t_ini=t_ini, t_fim=time.perf_counter())

        # 2) Acumular imagens por tracklet
        t_ini = time.perf_counter()
        tracklets_output_dir = os.path.join(run_path, "tracklets_images")
        save_images_by_tracklet(
            detections=detections,
            images_np=images_np,
            metadatas=metadatas,
            tracklet_images=self.tracklet_images_global,
            output_dir=tracklets_output_dir
        )
        log_timing('save_images_by_tracklet', t_ini=t_ini, t_fim=time.perf_counter())

        if self.JNRDebug:
            for track_id, imgs_list in self.tracklet_images_global.items():
                print(f"[tracklet_images_global][track {track_id}] contém {len(imgs_list)} imagens: {[fid for fid, _ in imgs_list]}")

        # 3) Para cada tracklet pronto, rodar pose+OCR
        t_ini_ocr = time.perf_counter()
        from concurrent.futures import ThreadPoolExecutor
        for track_id, imgs_list in list(self.tracklet_images_global.items()):
            t_tracklet_ini = time.perf_counter()
            if len(imgs_list) < self.jnr_batch_frames:
                continue

            cropped_list = []
            facing_away_flags = []
            def pose_job(args):
                frame_id, raw_img = args
                if raw_img is None or raw_img.size == 0 or self.pose_model is None:
                    return frame_id, None, None, raw_img
                t_pose_ini = time.perf_counter()
                keypoints, keypoints_scores = self.pose_infer_fn(self.pose_model, raw_img)
                t_pose_fim = time.perf_counter()
                log_timing('pose_inference', track_id, extra=frame_id, t_ini=t_pose_ini, t_fim=t_pose_fim)
                return frame_id, keypoints, keypoints_scores, raw_img

            with ThreadPoolExecutor(max_workers=4) as executor:
                pose_results = list(executor.map(pose_job, imgs_list))

            for frame_id, keypoints, keypoints_scores, raw_img in pose_results:
                if keypoints is None or keypoints_scores is None:
                    #log.warning(f"[DEBUG] Pose falhou para track {track_id}, frame {frame_id}. raw_img shape: {None if raw_img is None else raw_img.shape}")
                    continue
                try:
                    metrics = compute_pose_metrics(keypoints, keypoints_scores)
                    entry = {"track_id": track_id, "frame_id": frame_id}
                    entry.update(metrics)
                    self.pose_debug_data[track_id].append(entry)
                except Exception as e:
                    log.warning(f"[process] Falha ao computar métricas de pose para track {track_id}, frame {frame_id}: {e}")
                t_crop_ini = time.perf_counter()
                # --- Heurística do tronco vs direção do movimento ---
                ls = np.array(keypoints['left_shoulder'])
                rs = np.array(keypoints['right_shoulder'])
                lh = np.array(keypoints['left_hip'])
                rh = np.array(keypoints['right_hip'])
                mid_shoulder = (ls + rs) / 2
                mid_hip = (lh + rh) / 2
                trunk_vec = mid_hip - mid_shoulder
                # Salva trunk_vec para análise posterior
                self.trunk_motion_debug[track_id].append({
                    'frame_id': frame_id,
                    'mid_shoulder_x': float(mid_shoulder[0]),
                    'mid_shoulder_y': float(mid_shoulder[1]),
                    'mid_hip_x': float(mid_hip[0]),
                    'mid_hip_y': float(mid_hip[1]),
                    'trunk_vec_x': float(trunk_vec[0]),
                    'trunk_vec_y': float(trunk_vec[1]),
                })
                # Calcula e salva o angle_trunk_motion para este frame usando o motion_vec do tracklet
                entries = self.trunk_motion_debug[track_id]
                if len(entries) > 1:
                    dx = entries[-1]['mid_shoulder_x'] - entries[0]['mid_shoulder_x']
                    dy = entries[-1]['mid_shoulder_y'] - entries[0]['mid_shoulder_y']
                    motion_vec = np.array([dx, dy])
                    motion_norm = np.linalg.norm(motion_vec)
                    if motion_norm > 1e-3:
                        motion_vec = motion_vec / motion_norm
                        trunk_vec = np.array([entries[-1]['trunk_vec_x'], entries[-1]['trunk_vec_y']])
                        trunk_norm = np.linalg.norm(trunk_vec)
                        if trunk_norm > 1e-3:
                            trunk_unit = trunk_vec / trunk_norm
                            dot = np.dot(trunk_unit, motion_vec)
                            dot = np.clip(dot, -1.0, 1.0)
                            angle_trunk_motion = float(np.degrees(np.arccos(dot)))
                        else:
                            angle_trunk_motion = 0.0
                    else:
                        angle_trunk_motion = 0.0
                    self.trunk_motion_debug[track_id][-1]['angle_trunk_motion'] = angle_trunk_motion
                else:
                    self.trunk_motion_debug[track_id][-1]['angle_trunk_motion'] = 0.0
                # --- Lógica de costas com thresholds configuráveis ---
                is_away = is_facing_away(
                    keypoints, keypoints_scores,
                    angle_threshold_1=self.angle_threshold_1,
                    angle_threshold_2=self.angle_threshold_2,
                    angle_threshold_3=self.angle_threshold_3,
                    score_threshold_1=self.score_threshold_1,
                    score_threshold_2=self.score_threshold_2
                )
                facing_away_flags.append(is_away)
                if is_away:
                    back_img = crop_back_region(raw_img, keypoints)
                    debug_filename = os.path.join(run_path, f"debug_pose/track_{track_id}_frame_{frame_id}_back.jpg")
                    debug_save_image(
                        output_path=debug_filename,
                        image_rgb=back_img,
                        keypoints_dict=None,
                        debug_text="Costas detectadas (só crop)"
                    )
                    debug_filename_skel = os.path.join(run_path, f"debug_pose/track_{track_id}_frame_{frame_id}_skel.jpg")
                    debug_save_image(
                        output_path=debug_filename_skel,
                        image_rgb=raw_img,
                        keypoints_dict=keypoints,
                        skeleton=COCO_SKELETON,
                        debug_text="Costas detectadas (original + skeleton)"
                    )
                    cropped_list.append((frame_id, back_img))
                else:
                    debug_filename_front = os.path.join(run_path, f"debug_pose/track_{track_id}_frame_{frame_id}_front.jpg")
                    debug_save_image(
                        output_path=debug_filename_front,
                        image_rgb=raw_img,
                        keypoints_dict=keypoints,
                        skeleton=COCO_SKELETON,
                        debug_text="Frente detectada"
                    )
                t_crop_fim = time.perf_counter()
                log_timing('crop_back', track_id, extra=frame_id, t_ini=t_crop_ini, t_fim=t_crop_fim)

            # --- Percentual de costas usando heurística combinada (tronco vs movimento) ---
            percent_away = 0.0
            if len(self.trunk_motion_debug[track_id]) > 2:
                entries = self.trunk_motion_debug[track_id]
                dx = entries[-1]['mid_shoulder_x'] - entries[0]['mid_shoulder_x']
                dy = entries[-1]['mid_shoulder_y'] - entries[0]['mid_shoulder_y']
                motion_vec = np.array([dx, dy])
                motion_norm = np.linalg.norm(motion_vec)
                if motion_norm > 1e-3:
                    motion_vec = motion_vec / motion_norm
                else:
                    motion_vec = None
                facing_away_flags = []
                # Ignora o primeiro frame no cálculo do percentual
                for frame_idx, (frame_id, keypoints, keypoints_scores, raw_img) in list(enumerate(pose_results))[1:]:
                    if keypoints is None or keypoints_scores is None:
                        facing_away_flags.append(False)
                        continue
                    is_away = is_facing_away_trunk_motion(
                        keypoints, keypoints_scores, motion_vec=motion_vec,
                        angle_trunk_motion_threshold=self.angle_trunk_motion_threshold,
                        angle_threshold_1=self.angle_threshold_1,
                        angle_threshold_2=self.angle_threshold_2,
                        angle_threshold_3=self.angle_threshold_3,
                        score_threshold_1=self.score_threshold_1,
                        score_threshold_2=self.score_threshold_2
                    )
                    facing_away_flags.append(is_away)
                percent_away = 100.0 * sum(facing_away_flags) / len(facing_away_flags) if facing_away_flags else 0.0
            else:
                percent_away = 0.0
            if percent_away < self.min_percent_away:
                if self.JNRDebug:
                    print(f"[Tracklet {track_id}] {percent_away:.1f}% de costas < min_percent_away={self.min_percent_away}%. NÃO envia para OCR.")
                self.tracklet_images_global[track_id] = []
                continue
            if self.JNRDebug:
                print(f"[Tracklet {track_id}] {percent_away:.1f}% de costas >= min_percent_away={self.min_percent_away}%. Enviando para OCR.")
            # Agrupar para OCR
            cropped_list_rgb = [(track_id, fid, img) for fid, img in cropped_list]
            # Rodar OCR por tracklet
            t_ocr_ini = time.perf_counter()
            result_dict = self.run_paddleocr_batch(cropped_list_rgb)
            t_ocr_fim = time.perf_counter()
            log_timing('ocr_inference', track_id, extra=f'{len(cropped_list_rgb)} frames', t_ini=t_ocr_ini, t_fim=t_ocr_fim)

            batch_jn, batch_conf = result_dict.get(track_id, (None, 0.0))
            if batch_jn is not None and self.JNRDebug:
                print(f"[TrackletOCR] Track {track_id} → jersey {batch_jn} (conf={batch_conf:.2f})")

            frames_do_batch = [fid for (fid, _) in cropped_list]
            if batch_jn is not None:
                prev_jn, prev_conf = self.best_by_track.get(track_id, (None, 0.0))
                if self.should_replace(prev_jn, prev_conf, batch_jn, batch_conf, track_id=track_id):
                    self.best_by_track[track_id] = (batch_jn, batch_conf)
                    chosen_jn, chosen_conf = batch_jn, batch_conf
                else:
                    chosen_jn, chosen_conf = prev_jn, prev_conf
                mask = detections['image_id'].isin(frames_do_batch)
                detections.loc[mask, 'ocr_tracklet_pose_jn']   = chosen_jn
                detections.loc[mask, 'ocr_tracklet_pose_conf'] = chosen_conf
            # Buffer de blocos
            if batch_jn is not None:
                block_buffer = self.tracklet_detection_buffer[track_id]
                block_buffer.append(batch_jn)
            # Manter apenas último no buffer
            if imgs_list:
                self.tracklet_images_global[track_id] = [imgs_list[-1]]
            else:
                self.tracklet_images_global[track_id] = []
            t_tracklet_fim = time.perf_counter()
            log_timing('tracklet_total', track_id, t_ini=t_tracklet_ini, t_fim=t_tracklet_fim)

        log_timing('ocr+pose_por_tracklet', t_ini=t_ini_ocr, t_fim=time.perf_counter())

        # 4) Acumular comparação OCR direto vs tracklet+pose
        t_ini = time.perf_counter()
        for i in range(len(detections)):
            self.comparison_rows.append({
                "track_id": detections.iloc[i].get("track_id"),
                "frame_id": detections.iloc[i].get("image_id"),
                "ocr_direct_jn": detections.iloc[i].get("jersey_number_detection"),
                "ocr_direct_conf": detections.iloc[i].get("jersey_number_confidence"),
                "ocr_tracklet_pose_jn": detections.iloc[i].get("ocr_tracklet_pose_jn"),
                "ocr_tracklet_pose_conf": detections.iloc[i].get("ocr_tracklet_pose_conf")
            })
        log_timing('comparacao_ocr', t_ini=t_ini, t_fim=time.perf_counter())

        t3_end = time.perf_counter()
        log_timing('process_total', t_ini=t0_total, t_fim=t3_end)
        return detections

    @torch.no_grad()
    def run_superres(self, img_bgr: np.ndarray, identifier: str) -> np.ndarray:

        if not getattr(self, "upscaler", None):
            return img_bgr

        """Aplica Real-ESRGAN 4× quando a imagem é pequena (< 96 px)."""
        h, w = img_bgr.shape[:2]
        if not self.use_superres or h >= 96:
            return img_bgr                  # não precisa de upscaling

        # --- usa sempre o upscaler 4× pré-carregado ---
        upscaler = self.upscaler            # criado no __init__
        upscaler.tile = 128                 # opcional: ajusta tiling

        # ---- DEBUG opcional ----
        if self.JNRDebug:
            run_dir = os.getenv("HYDRA_RUN_DIR", os.getcwd())
            ddir = os.path.join(run_dir, "debug_sr_opt")
            os.makedirs(ddir, exist_ok=True)
            cv2.imwrite(os.path.join(ddir, f"{identifier}_in.png"), img_bgr)

        sr_bgr, _ = upscaler.enhance(img_bgr, outscale=4)

        # --- normaliza tipo & contiguidade ---
        if sr_bgr.dtype != np.uint8:
            sr_bgr = (np.clip(sr_bgr, 0., 1.) * 255).round().astype(np.uint8)
        if not sr_bgr.flags['C_CONTIGUOUS']:
            sr_bgr = np.ascontiguousarray(sr_bgr)

        if self.JNRDebug:
            cv2.imwrite(os.path.join(ddir, f"{identifier}_out.png"), sr_bgr)

        return sr_bgr

    def ensure_model_exists(self, model_path, download_url=None):
        """
        Garante que o modelo existe no caminho especificado. Faz download se necessário.
        """
        if not os.path.exists(model_path):
            if download_url is None:
                raise FileNotFoundError(f"Modelo não encontrado e download_url não fornecida: {model_path}")
            import requests
            print(f"Baixando modelo para {model_path} ...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            r = requests.get(download_url, allow_redirects=True)
            with open(model_path, 'wb') as f:
                f.write(r.content)
            print("Download concluído.")

    @torch.no_grad()
    def run_paddleocr_batch(self, track_images):
        if self.JNRDebug:
            print(">>> ENTER OCR batch processing")
        
        grouped = defaultdict(list)
        for track_id, frame_id, img in track_images:
            if img is not None and img.size:
                grouped[track_id].append((frame_id, img))

        results_by_track = {}

        for track_id, frames_list in grouped.items():
            if self.JNRDebug:
                print(f"  [Track {track_id}] frames_list:", [f for f, _ in frames_list])
            t0 = time.perf_counter()
            detections = []
            min_votes = max(1, math.ceil(len(frames_list) / 2))

            enh_imgs_bgr = []
            fid_order = []

            # --- SR + convert
            for fid, img_bgr in frames_list:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_enh_rgb = self.run_superres(img_rgb, f"track_{track_id}_frame_{fid}")
                img_enh_bgr = cv2.cvtColor(img_enh_rgb, cv2.COLOR_RGB2BGR)
                enh_imgs_bgr.append(img_enh_bgr)
                fid_order.append(fid)
                if self.JNRDebug:
                    print(f"    [Timing] SR frame {fid} levou {time.perf_counter()-t0:.3f}s")

            # Resultados de todos os modelos
            all_model_results = defaultdict(list)
            
            for idx, img_bgr in enumerate(enh_imgs_bgr):
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if self.imageDebug:
                    debug_dir = os.path.join(os.getenv("HYDRA_RUN_DIR","."), "debug_ocr_input")
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir, f"track{track_id}_frame{fid_order[idx]}_ocrinput.png"), img_rgb)
                # --- OTIMIZAÇÃO: rodar modelos em paralelo se não for ocr_only_model ---
                if self.ocr_only_model:
                    # Só roda o modelo escolhido
                    number, conf = self.run_ocr_with_model(img_rgb, self.ocr_only_model, track_id=track_id, frame_id=fid_order[idx])
                    if number and conf >= self.confidence_threshold:
                        all_model_results[self.ocr_only_model].append((number, conf))
                        if self.JNRDebug:
                            print(f"[{self.ocr_only_model}] detected {number} (conf={conf:.2f})")
                else:
                    import concurrent.futures
                    def ocr_task(model_name):
                        return model_name, self.run_ocr_with_model(img_rgb, model_name, track_id=track_id, frame_id=fid_order[idx])
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(ocr_task, model_name) for model_name in self.ocr_models.keys()]
                        for future in concurrent.futures.as_completed(futures):
                            model_name, (number, conf) = future.result()
                            if number and conf >= self.confidence_threshold:
                                all_model_results[model_name].append((number, conf))
                                if self.JNRDebug:
                                    print(f"[{model_name}] detected {number} (conf={conf:.2f})")

            # Votação considerando todos os modelos
            final_number, final_conf = self.ensemble_voting(all_model_results)
            if final_number:
                results_by_track[track_id] = (final_number, final_conf)

        return results_by_track

    def ensemble_voting(self, all_results):
        """
        Votação entre diferentes modelos
        """
        all_detections = []
        for model_name, detections in all_results.items():
            for number, conf in detections:
                all_detections.append((number, conf, model_name))

        if not all_detections:
            return None, 0.0

        # Conta ocorrências de cada número
        number_counts = Counter(n for n, _, _ in all_detections)
        
        # Pega o número mais votado
        most_common = number_counts.most_common(1)
        if not most_common:
            return None, 0.0

        number = most_common[0][0]
        # Média das confianças para este número
        confs = [c for n, c, _ in all_detections if n == number]
        avg_conf = sum(confs) / len(confs)

        return number, avg_conf

    @torch.no_grad()
    def finalize_ocr(self, detections: pd.DataFrame) -> pd.DataFrame:
        run_path = os.getenv("HYDRA_RUN_DIR", os.getcwd())

        # 1) Processa quaisquer tracklets que ainda tenham frames pendentes
        for track_id, frames_list in list(self.tracklet_images_global.items()):
            if not frames_list:
                continue
            # roda OCR nesses frames pendentes
            batch = [(track_id, fid, img) for fid, img in frames_list]
            result_dict = self.run_paddleocr_batch(batch)
            final_jn, final_conf = result_dict.get(track_id, (None, 0.0))
            prev_jn, prev_conf = self.best_by_track.get(track_id, (None, 0.0))
            # usa should_replace para decidir atualização
            if self.should_replace(prev_jn, prev_conf, final_jn, final_conf, track_id=track_id):
                self.best_by_track[track_id] = (final_jn, final_conf)

        # 2) Zera colunas de saída e preenche com o melhor jersey por track
        detections['jersey_number_detection']  = None
        detections['jersey_number_confidence'] = 0.0

        # --- NOVO: decisão final baseada no buffer de confiança ---
        decision_log = []  # Para auditoria
        for track_id in set(list(self.best_by_track.keys()) + list(self.confidence_sum_buffer.keys())):
            if track_id is None:
                continue
            if self.ocr_decision_strategy == "confidence_sum":
                conf_dict = {str(k): float(v) for k, v in self.confidence_sum_buffer[track_id].items() if k is not None and k != ''}
                if conf_dict:
                    # Frequência de cada número no buffer de blocos (robustez real)
                    block_buffer = list(self.tracklet_detection_buffer[track_id])
                    det_freq = {}
                    for num in block_buffer:
                        det_freq[num] = det_freq.get(num, 0) + 1
                    # Loga todas as somas para auditoria
                    for num, soma in conf_dict.items():
                        decision_log.append({"track_id": track_id, "number": num, "confidence_sum": soma, "freq": det_freq.get(num, 0)})
                    best_num = max(conf_dict, key=lambda k: conf_dict[k])
                    freq = det_freq.get(best_num, 0)
                    # Só aceita se best_num aparece min_votes vezes seguidas no buffer de blocos
                    if freq >= self.min_votes and self.has_min_consecutive(block_buffer, best_num, self.min_votes):
                        max_conf = 0.0
                        for logs in self.tracklet_detection_logs.get(track_id, []):
                            if logs.get('number') == best_num:
                                max_conf = max(max_conf, logs.get('confidence', 0.0))
                        self.best_by_track[track_id] = (best_num, max_conf)
                    else:
                        self.best_by_track[track_id] = (None, 0.0)
            best_jn, best_conf = self.best_by_track.get(track_id, (None, 0.0))
            if best_jn is None or track_id is None:
                continue
            mask = detections['track_id'] == track_id
            detections.loc[mask, 'jersey_number_detection']  = best_jn
            detections.loc[mask, 'jersey_number_confidence'] = best_conf

        # Salva log detalhado da decisão do confidence_sum
        if self.ocr_decision_strategy == "confidence_sum" and decision_log:
            log_dir = os.path.join(run_path, "ocr_confidence_sum_decision_logs")
            os.makedirs(log_dir, exist_ok=True)
            df_dec = pd.DataFrame(decision_log)
            out_path = os.path.join(log_dir, "confidence_sum_decision_log.csv")
            df_dec.to_csv(out_path, index=False)
            log.info(f"📄 Log de decisão do confidence_sum exportado para {out_path}")

        # 3) (Opcional) Limpa o buffer de tracklets
        self.tracklet_images_global.clear()

        # 4) Exporta CSV “por track” baseado em best_by_track
        rows = [
            {"track_id": tid,
            "jersey_number_detection": jn,
            "jersey_number_confidence": conf}
            for tid, (jn, conf) in sorted(self.best_by_track.items())
            if jn is not None and tid is not None
        ]
        if rows:
            df = pd.DataFrame(rows)
            out_csv = os.path.join(run_path, "ocr_results_tracklet_pose_all.csv")
            df.to_csv(out_csv, index=False)
            log.info(f"📄 Resultados OCR por track exportados para: {out_csv}")
        else:
            log.info("⚠️ Nenhum jersey detectado para exportar em ocr_results_tracklet_pose_all.csv")

        # 5) Exporta a comparação OCR-direto vs tracklet+pose (caso tenha sido usada em process)
        try:
            self.export_comparison_csv()
        except Exception as e:
            log.warning(f"[finalize_ocr] Falha ao exportar comparação OCR: {e}")

        # 6) Exporta logs de debug de OCR, se houver
        if self.debug_paddleocr and self.tracklet_debug_data:
            debug_dir = os.path.join(run_path, "debug_ocr")
            os.makedirs(debug_dir, exist_ok=True)
            # debug por frame
            try:
                df_dbg = pd.DataFrame([
                    {"track_id": tid, **entry}
                    for tid, lst in self.tracklet_debug_data.items()
                    for entry in lst
                ])
                path1 = os.path.join(debug_dir, "ocr_debug_per_frame.csv")
                df_dbg.to_csv(path1, index=False)
                log.info(f"📄 OCR debug per frame salvo em: {path1}")
            except Exception as e:
                log.warning(f"[finalize_ocr] Falha ao salvar ocr_debug_per_frame: {e}")

            # resumo por track
            summary = []
            for tid, lst in self.tracklet_debug_data.items():
                df_tmp = pd.DataFrame(lst).dropna(subset=['jersey_number','confidence'])
                if df_tmp.empty:
                    continue
                mode = df_tmp['jersey_number'].mode()
                if mode.empty:
                    continue
                num = mode.iat[0]
                conf = df_tmp[df_tmp['jersey_number']==num]['confidence'].mean()
                summary.append({"track_id": tid, "final_jersey_number": num, "avg_confidence": conf})
            if summary:
                try:
                    df_sum = pd.DataFrame(summary)
                    path2 = os.path.join(debug_dir, "ocr_debug_resumo_por_track.csv")
                    df_sum.to_csv(path2, index=False)
                    log.info(f"📄 OCR debug resumo por track salvo em: {path2}")
                except Exception as e:
                    log.warning(f"[finalize_ocr] Falha ao salvar ocr_debug_resumo_por_track: {e}")

        # 7) Exporta métricas de pose coletadas em process()
        #   Supondo que em self.pose_debug_data você acumulou dicts {"track_id", "frame_id", ... métricas ...}
        if self.JNRDebug:
            try:
                from sn_gamestate.jersey.tracklets_utils import save_pose_debug_data
                # salva CSVs em run_path/debug_pose_metrics/
                save_pose_debug_data(self.pose_debug_data, run_path, prefix="pose_metrics")
                # limpa buffer de pose
                self.pose_debug_data.clear()
            except Exception as e:
                log.warning(f"[finalize_ocr] Falha ao salvar métricas de pose: {e}")

        # 8) Por fim, limpa best_by_track e debug data de OCR
        self.best_by_track.clear()
        self.tracklet_debug_data.clear()

        # 9) Exporta logs detalhados de detecção por track/modelo
        run_path = os.getenv("HYDRA_RUN_DIR", os.getcwd())
        log_dir = os.path.join(run_path, "ocr_detections_by_track")
        os.makedirs(log_dir, exist_ok=True)
        for track_id, logs in self.tracklet_detection_logs.items():
            df = pd.DataFrame(logs)
            out_path = os.path.join(log_dir, f"track_{track_id}_ocr_log.csv")
            df.to_csv(out_path, index=False)
        log.info(f"📄 Logs detalhados de OCR exportados para {log_dir}")
        self.tracklet_detection_logs.clear()
        self.tracklet_detection_buffer.clear()

        # 10) Exporta buffer de confiança para auditoria (apenas se houver)
        if self.confidence_sum_buffer:
            conf_dir = os.path.join(run_path, "ocr_confidence_sum_logs")
            os.makedirs(conf_dir, exist_ok=True)
            for track_id, conf_dict in self.confidence_sum_buffer.items():
                df = pd.DataFrame([
                    {"number": k, "confidence_sum": v}
                    for k, v in conf_dict.items()
                ])
                out_path = os.path.join(conf_dir, f"track_{track_id}_confidence_sum.csv")
                df.to_csv(out_path, index=False)
            log.info(f"📄 Buffers de soma de confiança exportados para {conf_dir}")
            self.confidence_sum_buffer.clear()

        # Após processar todos os tracklets, exporta CSVs de orientação do tronco para análise
        if self.trunk_motion_debug:
            run_path = os.getenv("HYDRA_RUN_DIR", os.getcwd())
            trunk_dir = os.path.join(run_path, "debug_trunk_motion")
            os.makedirs(trunk_dir, exist_ok=True)
            for track_id, entries in self.trunk_motion_debug.items():
                df = pd.DataFrame(entries)
                out_file = os.path.join(trunk_dir, f"trunk_motion_track_{track_id}.csv")
                df.to_csv(out_file, index=False)
                log.info(f"Trunk motion debug salvo: {out_file}")
            self.trunk_motion_debug.clear()

        return detections

    def export_comparison_csv(self, path="comparacao_ocr_pose.csv"):
        """
        Exporta a comparação entre OCR direto e OCR com tracklet+pose para CSV.
        """
        if not self.comparison_rows:
            print("⚠️ Nenhuma comparação foi acumulada.")
            return
        run_path = os.getenv("HYDRA_RUN_DIR", os.getcwd())
        output_file = os.path.join(run_path, path)
        df = pd.DataFrame(self.comparison_rows)
        df.to_csv(output_file, index=False)
        if self.JNRDebug:
            print(f"📄 Comparação OCR salva em: {path}")

    def extract_jersey_numbers_from_paddleocr(self, ocr_result):
        """
        Pode receber:
        - List[List[(box, (text, score))]]  (antigo formato)
        - List[(box, (text, score))]        (det=False, rec=True, cls=False)
        """
        # 1) nada detectado
        if not ocr_result:
            return None, 0.0

        # 2) achata se vierem listas aninhadas
        elems = ocr_result[0] if isinstance(ocr_result[0], list) else ocr_result

        digit_boxes = []
        for entry in elems:
            # valida que entry é um tuple/list de tamanho 2
            if (
                not isinstance(entry, (tuple, list)) or
                len(entry) != 2 or
                not isinstance(entry[0], (tuple, list)) or
                not isinstance(entry[1], (tuple, list)) or
                len(entry[1]) != 2
            ):
                continue

            box, (text, score) = entry
            txt = text.strip()

            # só números de 1 ou 2 dígitos
            if re.fullmatch(r'\d{1,2}', txt):
                # altura da caixa em pixels
                height = max(p[1] for p in box) - min(p[1] for p in box)
                digit_boxes.append((txt, score, height))

        # se nada válido, retorna None
        if not digit_boxes:
            return None, 0.0

        # 3) escolhe o dígito com maior altura
        best_txt, best_score, _ = max(digit_boxes, key=lambda x: x[2])
        return best_txt, best_score
    
    def update_majority_buffer(self, track_id, batch_jn, batch_conf):
        N = self.jnr_batch_frames
        det_buffer = self.tracklet_detection_buffer[track_id]
        det_buffer.append(batch_jn)
        if len(det_buffer) > N:
            det_buffer.pop(0)
        count = Counter([n for n in det_buffer if n])
        if count:
            most_common, freq = count.most_common(1)[0]
        else:
            most_common, freq = None, 0
        return most_common, freq

    def should_replace(self, prev_jn, prev_conf, batch_jn, batch_conf, track_id=None, frame_id=None):
        """
        Estratégia selecionável: sliding_persistence ou confidence_sum
        """
        if self.ocr_decision_strategy == "confidence_sum":
            # Acumula confiança por número, descartando None/nulo
            if self.JNRDebug:
                print(f"[should_replace][confidence_sum] track_id={track_id} batch_jn={batch_jn} batch_conf={batch_conf}")
            if batch_jn is not None:
                self.confidence_sum_buffer[track_id][batch_jn] += batch_conf
            conf_dict = {k: v for k, v in self.confidence_sum_buffer[track_id].items() if k is not None and k != ''}
            if self.JNRDebug:
                print(f"[should_replace][confidence_sum] conf_dict={conf_dict}")
            if not conf_dict:
                return False
            best_num = max(conf_dict, key=lambda k: conf_dict[k])
            if self.JNRDebug:
                print(f"[confidence_sum][track {track_id}] confs: {dict(conf_dict)} | best: {best_num}")
            return best_num != prev_jn
        else:
            # Sliding persistence: verifica se o número atual é igual ao anterior
            return batch_jn != prev_jn

    def run_ocr_with_model(self, img, model_name, track_id=None, frame_id=None, log_all=True):
        model = self.ocr_models.get(model_name)
        if model is None:
            if log_all and track_id is not None and frame_id is not None:
                self.tracklet_detection_logs[track_id].append({
                    'frame_id': frame_id,
                    'model': model_name,
                    'number': None,
                    'confidence': 0.0
                })
            return None, 0.0
        number, conf = None, 0.0
        try:
            if model_name.startswith('ppocr'):
                result = model.ocr(img, det=True, rec=True, cls=False)
                number, conf = self.extract_jersey_numbers_from_paddleocr(result)
            elif model_name == 'easyocr':
                result = model.readtext(img)
                digit_results = [(re.sub(r'\D', '', text), conf) for _, text, conf in result if any(c.isdigit() for c in text)]
                digit_results = [(text[:2], conf) for text, conf in digit_results if 1 <= len(text[:2]) <= 2]
                if digit_results:
                    best_text, best_conf = max(digit_results, key=lambda x: x[1])
                    number, conf = best_text, float(best_conf)
        except Exception as e:
            if self.JNRDebug:
                print(f"Erro no modelo {model_name}: {e}")
        if log_all and track_id is not None and frame_id is not None:
            self.tracklet_detection_logs[track_id].append({
                'frame_id': frame_id,
                'model': model_name,
                'number': number,
                'confidence': conf
            })
        return number, conf

    def has_min_consecutive(self, buffer, value, min_consecutive):
        """
        Verifica se 'value' aparece pelo menos 'min_consecutive' vezes seguidas em 'buffer'.
        """
        count = 0
        for v in buffer:
            if v == value:
                count += 1
                if count >= min_consecutive:
                    return True
            else:
                count = 0
        return False

    def calculate_trunk_motion_direction(self, track_id):
        """
        Calcula a direção do movimento do tronco para o track_id especificado.
        Adiciona as entradas 'motion_vec_x', 'motion_vec_y' e 'angle_trunk_motion' para cada frame no debug.
        """
        if len(self.trunk_motion_debug[track_id]) > 1:
            entries = self.trunk_motion_debug[track_id]
            # Direção do movimento: diferença entre mid_shoulder do último e do primeiro frame
            dx = entries[-1]['mid_shoulder_x'] - entries[0]['mid_shoulder_x']
            dy = entries[-1]['mid_shoulder_y'] - entries[0]['mid_shoulder_y']
            motion_vec = np.array([dx, dy])
            motion_norm = np.linalg.norm(motion_vec)
            if motion_norm > 1e-3:
                motion_vec = motion_vec / motion_norm
            else:
                motion_vec = np.zeros(2)
            # Para cada frame, calcula ângulo entre trunk_vec e motion_vec
            for entry in entries:
                trunk_vec = np.array([entry['trunk_vec_x'], entry['trunk_vec_y']])
                trunk_norm = np.linalg.norm(trunk_vec)
                if trunk_norm > 1e-3:
                    trunk_unit = trunk_vec / trunk_norm
                    dot = np.dot(trunk_unit, motion_vec)
                    dot = np.clip(dot, -1.0, 1.0)
                    angle_trunk_motion = float(np.degrees(np.arccos(dot)))
                else:
                    angle_trunk_motion = np.nan
                entry['motion_vec_x'] = float(motion_vec[0])
                entry['motion_vec_y'] = float(motion_vec[1])
                entry['angle_trunk_motion'] = angle_trunk_motion
                # Salva também o angle_trunk_motion no trunk_motion_debug usando o motion_vec correto do tracklet
                if len(self.trunk_motion_debug[track_id]) > 1:
                    entries = self.trunk_motion_debug[track_id]
                    dx = entries[-1]['mid_shoulder_x'] - entries[0]['mid_shoulder_x']
                    dy = entries[-1]['mid_shoulder_y'] - entries[0]['mid_shoulder_y']
                    motion_vec = np.array([dx, dy])
                    motion_norm = np.linalg.norm(motion_vec)
                    if motion_norm > 1e-3:
                        motion_vec = motion_vec / motion_norm
                        for i, entry in enumerate(entries):
                            trunk_vec = np.array([entry['trunk_vec_x'], entry['trunk_vec_y']])
                            trunk_norm = np.linalg.norm(trunk_vec)
                            if trunk_norm > 1e-3:
                                trunk_unit = trunk_vec / trunk_norm
                                dot = np.dot(trunk_unit, motion_vec)
                                dot = np.clip(dot, -1.0, 1.0)
                                angle_trunk_motion = float(np.degrees(np.arccos(dot)))
                            else:
                                angle_trunk_motion = 0.0
                            self.trunk_motion_debug[track_id][i]['angle_trunk_motion'] = angle_trunk_motion
                    else:
                        for i, entry in enumerate(entries):
                            self.trunk_motion_debug[track_id][i]['angle_trunk_motion'] = 0.0
                else:
                    self.trunk_motion_debug[track_id][-1]['angle_trunk_motion'] = 0.0
