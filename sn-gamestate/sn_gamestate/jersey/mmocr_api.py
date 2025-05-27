import pandas as pd
import torch
import numpy as np
import os ,psutil
import math
import logging
import cv2
import subprocess
import re
from collections import defaultdict, Counter

from paddleocr import PaddleOCR
import gc, paddle, torch

from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.utils import bbox2poly, crop_img, poly2bbox

from mmengine.registry import init_default_scope
init_default_scope('mmpose')

from mmpose.apis import MMPoseInferencer

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from sn_gamestate.jersey.mmocr_utils import (
    run_hrnet_pose_inference,
    save_ocr_results,
    alternative_jersey_number_detection,
    save_images_by_tracklet,
    is_facing_away,
    crop_back_region,
    debug_save_image,
    COCO_SKELETON
)


log = logging.getLogger(__name__)

class MMOCR(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, batch_size, device, **kwargs):
        super().__init__(batch_size=batch_size)
        self.device = device
        self.batch_size = batch_size
        self.jnr_batch_frames = kwargs.get("jnr_batch_frames", 5)
        self.use_superres = kwargs.get("use_superres", True)
        self.tracklet_images_global = {}
        self.comparison_rows = []
        self.debug_paddleocr = kwargs.get("debug_paddleocr", True)
        self.tracklet_debug_data = defaultdict(list) 

        # MMOCR baseline
        self.textdetinferencer = TextDetInferencer('dbnet_resnet18_fpnc_1200e_icdar2015', device=device)
        self.textrecinferencer = TextRecInferencer('SAR', device=device)

        # Pose model
        self.pose_model = kwargs.get("pose_model") or MMPoseInferencer(
            pose2d='human',
            det_model='yolox_l_8x8_300e_coco',
            det_cat_ids=[0],
        )

        # PaddleOCR para tracklets
        ocr_root = "/home/joao/soccernet/pretrained_models/paddleocr"
        self.paddle_ocr = PaddleOCR(
            use_angle_cls=True, lang='en', show_log=False, use_gpu=True,
            det_model_dir=os.path.join(ocr_root, "en_PP-OCRv3_det_infer"),
            rec_model_dir=os.path.join(ocr_root, "en_PP-OCRv4_rec_infer"),
            cls_model_dir=os.path.join(ocr_root, "ch_ppocr_mobile_v2.0_cls_infer"),
        )

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

        images_np = [img.cpu().numpy() for img in batch['img']]
        del batch['img']

        # 1) OCR direto frame a frame
        predictions = self.run_mmocr_inference(images_np)
        ocr_direct_jn = []
        ocr_direct_conf = []
        for prediction in predictions:
            jn, conf = self.extract_jersey_numbers_from_ocr(prediction)
            ocr_direct_jn.append(jn)
            ocr_direct_conf.append(conf)

        # Preenche os campos esperados pela baseline
        detections['jersey_number_detection'] = ocr_direct_jn
        detections['jersey_number_confidence'] = ocr_direct_conf

        # 2) Salvar CSV parcial
        #output_file = os.path.join(run_path, "ocr_results_direct.csv")
        #save_ocr_results(detections, output_file=output_file)

        # 3) Acumular imagens por tracklet
        tracklets_output_dir = os.path.join(run_path, "tracklets_images")
        save_images_by_tracklet(
            detections=detections,
            images_np=images_np,
            metadatas=metadatas,
            tracklet_images=self.tracklet_images_global,
            output_dir=tracklets_output_dir
        )

        # 4) Processar tracklets acumulados com pose + OCR
        for track_id, imgs_list in self.tracklet_images_global.items():
            if len(imgs_list) >= self.jnr_batch_frames:
                cropped_list = []
                for frame_id, raw_img in imgs_list:
                    if raw_img is None or raw_img.size == 0 or self.pose_model is None:
                        cropped_list.append((frame_id, raw_img))
                        continue

                    keypoints, keypoints_scores = run_hrnet_pose_inference(self.pose_model, raw_img)
                    if keypoints is None or keypoints_scores is None:
                        cropped_list.append((frame_id, raw_img))
                        continue

                    if is_facing_away(keypoints, keypoints_scores):
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
                        cropped_list.append((frame_id, raw_img))

                cropped_list_rgb = []
                for fid, img in cropped_list:
                    if img is None or img.size == 0:
                        cropped_list_rgb.append((track_id, fid, img))
                    else:
                        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        cropped_list_rgb.append((track_id, fid, img))

                print(f"run_paddleocr_batch")
                result_dict = self.run_paddleocr_batch(cropped_list_rgb)
                if track_id in result_dict:
                    best_jn, best_conf = result_dict[track_id]
                else:
                    best_jn, best_conf = None, 0.0

                # Marcar todos os frames com o resultado agregado
                mask = detections['track_id'] == track_id
                detections.loc[mask, 'ocr_tracklet_pose_jn'] = best_jn
                detections.loc[mask, 'ocr_tracklet_pose_conf'] = best_conf
                if imgs_list:                               # mantém só o último frame do lote
                    self.tracklet_images_global[track_id] = [imgs_list[-1]]
                else:
                    self.tracklet_images_global[track_id] = []

        # 5) Adicionar na comparison_rows
        for i in range(len(detections)):
            self.comparison_rows.append({
                "track_id": detections.iloc[i].get("track_id"),
                "frame_id": detections.iloc[i].get("image_id"),
                "ocr_direct_jn": ocr_direct_jn[i],
                "ocr_direct_conf": ocr_direct_conf[i],
                "ocr_tracklet_pose_jn": detections.iloc[i].get("ocr_tracklet_pose_jn"),
                "ocr_tracklet_pose_conf": detections.iloc[i].get("ocr_tracklet_pose_conf")
            })

        return detections


    @torch.no_grad()
    def run_superres(self, img_rgb: np.ndarray, identifier: str) -> np.ndarray:
        """
        Faz super-resolução com Real-ESRGAN.

        Args
        ----
        img_rgb   : imagem RGB uint8 (H×W×3, 0-255)
        identifier: string curta para nomear ficheiros de debug

        Returns
        -------
        sr_rgb    : imagem RGB uint8 (0-255)
        """

        if not self.use_superres:
            return img_rgb  # early-exit

        # 0)  garanta que o identificador não cria sub-pastas nem sobrescreve ficheiros
        identifier = identifier.replace(os.sep, "_")

        # 1)  pasta de debug
        run_dir   = os.getenv("HYDRA_RUN_DIR", os.getcwd())
        debug_dir = os.path.join(run_dir, "debug_sr")
        os.makedirs(debug_dir, exist_ok=True)

        # 2)  salva entrada (RGB → BGR)
        in_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(os.path.join(debug_dir, f"{identifier}_in.png"), in_bgr)

        # 3)  SR (RealESRGANer espera BGR uint8)
        #     em versões ≥0.3.0 já devolve uint8; em <0.3.0 devolve float32 [0-1]
        sr_bgr, _ = self.upscaler.enhance(in_bgr, outscale=getattr(self.upscaler, "scale", 4))

        if sr_bgr.dtype != np.uint8:                      # compatibilidade com versões antigas
            sr_bgr = (np.clip(sr_bgr, 0.0, 1.0) * 255).round().astype(np.uint8)

        # 4)  salva saída (continua em BGR)
        #cv2.imwrite(os.path.join(debug_dir, f"{identifier}_out.png"), sr_bgr)

        # 5)  devolve em RGB
        sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)

        # 6)  sanity-check
        assert sr_rgb.dtype == np.uint8 and sr_rgb.min() >= 0 and sr_rgb.max() <= 255

        return sr_rgb


    def run_paddleocr_batch(self, track_images, confidence_threshold=0.70):
        """
        Processa um batch de imagens. Para cada track_id, devolve o número mais frequente
        (com média de confiança), apenas se aparecer em pelo menos 'min_votes' imagens válidas.
        """

        num_frames = len(frames)
        min_votes  = max(1, math.ceil(num_frames / 2))
        grouped = defaultdict(list)
        for track_id, frame_id, img in track_images:
            if img is not None and img.size != 0:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                grouped[track_id].append((frame_id, img_rgb))

        results_by_track = {}
        debug_dir = None

        if self.debug_paddleocr:
            run_path = os.getenv("HYDRA_RUN_DIR", os.getcwd())
            debug_dir = os.path.join(run_path, "debug_ocr_annotated")
            os.makedirs(debug_dir, exist_ok=True)

        for track_id, frames in grouped.items():
            detections = []

            for frame_id, img in frames:
                img_enh = self.run_superres(img, f"track_{track_id}_frame_{frame_id}")
                result = self.paddle_ocr.ocr(img_enh[:, :, ::-1], cls=True)

                jersey_number, confidence = self.extract_jersey_numbers_from_paddleocr(result)

                if jersey_number is not None and confidence >= confidence_threshold:
                    detections.append((jersey_number, confidence))

                # DEBUG visual
                if self.debug_paddleocr:
                    print(f"debug paddle ocr-- gravar imgs")
                    annotated = cv2.cvtColor(img_enh, cv2.COLOR_RGB2BGR)
                    if result and isinstance(result, list) and result[0]:
                        for r in result[0]:
                            box = np.array(r[0], dtype=np.int32)
                            text, score = r[1]
                            color = (0, 255, 0) if score >= confidence_threshold else (0, 0, 255)
                            cv2.polylines(annotated, [box], isClosed=True, color=color, thickness=2)
                            cv2.putText(annotated, f"{text} ({score:.2f})", tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    out_path = os.path.join(debug_dir, f"track_{track_id}_frame_{frame_id}_annotated.jpg")
                    cv2.imwrite(out_path, annotated)

                    # Salvar dados brutos
                    self.tracklet_debug_data[track_id].append({
                        "frame_id": frame_id,
                        "jersey_number": jersey_number,
                        "confidence": confidence
                    })

            if not detections:
                continue

            count = Counter([jn for jn, _ in detections])
            # Apenas números com pelo menos min_votes
            valid_candidates = [jn for jn, c in count.items() if c >= min_votes]
            if not valid_candidates:
                continue

            candidate_confidences = {
                jn: [conf for j, conf in detections if j == jn] for jn in valid_candidates
            }
            best_jn = max(candidate_confidences.items(), key=lambda item: sum(item[1]) / len(item[1]))[0]
            best_conf_mean = sum(candidate_confidences[best_jn]) / len(candidate_confidences[best_jn])

            results_by_track[track_id] = (best_jn, best_conf_mean)

        return results_by_track

    def run_mmocr_inference(self, images_np):
        """
        Roda detecção e reconhecimento de texto do MMOCR em mini batches.
        """
        result = {'det': [], 'rec': []}

        batch_size = self.batch_size

        for i in range(0, len(images_np), batch_size):
            batch_imgs = images_np[i:i+batch_size]

            det_preds = self.textdetinferencer(
                batch_imgs,
                return_datasamples=True,
                batch_size=batch_size,
                progress_bar=False,
            )['predictions']
            result['det'].extend(det_preds)

            for img, det_data_sample in zip(batch_imgs, det_preds):
                det_pred = det_data_sample.pred_instances
                rec_inputs = []
                for polygon in det_pred['polygons']:
                    quad = bbox2poly(poly2bbox(polygon)).tolist()
                    rec_input = crop_img(img, quad)
                    if rec_input.shape[0] == 0 or rec_input.shape[1] == 0:
                        continue
                    rec_inputs.append(rec_input)

                if rec_inputs:
                    rec_preds = self.textrecinferencer(
                        rec_inputs,
                        return_datasamples=True,
                        batch_size=batch_size,
                        progress_bar=False
                    )['predictions']
                else:
                    rec_preds = []

                result['rec'].append(rec_preds)

        # Simplifica resultados
        pred_results = [{} for _ in range(len(result['rec']))]
        for i, rec_pred_list in enumerate(result['rec']):
            result_out = dict(rec_texts=[], rec_scores=[])
            for rec_pred_instance in rec_pred_list:
                rec_dict_res = self.textrecinferencer.pred2dict(rec_pred_instance)
                result_out['rec_texts'].append(rec_dict_res['text'])
                result_out['rec_scores'].append(rec_dict_res['scores'])
            pred_results[i].update(result_out)

        return pred_results

    def finalize_ocr(self, detections: pd.DataFrame) -> pd.DataFrame:
        """
        Processa frames pendentes, agrega resultados por track e exporta CSVs.
        Devolve o DataFrame `detections` já preenchido.
        """
        run_path = os.getenv("HYDRA_RUN_DIR", os.getcwd())

        # ────────────────────────────────────────────────────────────────
        # 1. Faz OCR nos tracklets que ainda têm frames “pendentes”
        # ────────────────────────────────────────────────────────────────
        for track_id, frames_list in list(self.tracklet_images_global.items()):
            if not frames_list:          # lista vazia → nada a fazer
                continue

            # prepara imagens RGB (ou None) para o batch
            batch_imgs = [
                (track_id, fid, None if img is None or img.size == 0 else
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                for fid, img in frames_list
            ]

            result_dict = self.run_paddleocr_batch(batch_imgs)
            best_jn, best_conf = result_dict.get(track_id, (None, 0.0))

            mask = detections['track_id'] == track_id
            detections.loc[mask, 'ocr_tracklet_pose_jn']  = best_jn
            detections.loc[mask, 'ocr_tracklet_pose_conf'] = best_conf

        # esvazia a cache para o próximo vídeo
        self.tracklet_images_global.clear()

        # ────────────────────────────────────────────────────────────────
        # 2. Constrói o CSV com TODOS os tracks
        # ────────────────────────────────────────────────────────────────
        tracklet_pose_results = (
            detections[['track_id', 'ocr_tracklet_pose_jn', 'ocr_tracklet_pose_conf']]
            .sort_values(['track_id', 'ocr_tracklet_pose_conf'], ascending=[True, False])
            .drop_duplicates(subset=['track_id'])                     # mantém pares únicos track/número/conf
            .sort_values(['track_id', 'ocr_tracklet_pose_jn'])
        )

        out_csv = os.path.join(run_path, "ocr_results_tracklet_pose.csv")
        tracklet_pose_results.to_csv(out_csv, index=False)
        log.info(f"📄 Resultados OCR por tracklet exportados para: {out_csv}")

        # ────────────────────────────────────────────────────────────────
        # 3. Comparação OCR-direto  vs  tracklet+pose
        # ────────────────────────────────────────────────────────────────
        self.export_comparison_csv()

        # ────────────────────────────────────────────────────────────────
        # 4. Dumps de debug (opcional)
        # ────────────────────────────────────────────────────────────────
        if self.debug_paddleocr and self.tracklet_debug_data:
            debug_dir = os.path.join(run_path, "debug_ocr")
            os.makedirs(debug_dir, exist_ok=True)

            # a) registo por-frame
            pd.DataFrame([
                {"track_id": tid, **entry}
                for tid, lst in self.tracklet_debug_data.items()
                for entry in lst
            ]).to_csv(os.path.join(debug_dir, "ocr_debug_per_frame.csv"), index=False)

            # b) resumo por track
            summary_rows = []
            for tid, lst in self.tracklet_debug_data.items():
                df = pd.DataFrame(lst)
                if not df.empty:
                    num = df['jersey_number'].mode().iat[0]    # mais frequente
                    conf = df.loc[df['jersey_number'] == num, 'confidence'].mean()
                    summary_rows.append({"track_id": tid, "final_jersey_number": num,
                                        "avg_confidence": conf})
            if summary_rows:
                pd.DataFrame(summary_rows).to_csv(
                    os.path.join(debug_dir, "ocr_summary.csv"), index=False)

        # limpa buffers de debug para o próximo vídeo
        self.tracklet_debug_data.clear()

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
        print(f"📄 Comparação OCR salva em: {path}")

    def extract_jersey_numbers_from_paddleocr(self, ocr_result):
        if not ocr_result or not isinstance(ocr_result, list) or not ocr_result[0]:
            return None, 0

        digit_boxes = []
        for r in ocr_result[0]:
            text, score = r[1]
            digits = re.fullmatch(r'\d{1,2}', text.strip())
            if digits:
                box = r[0]
                height = max(p[1] for p in box) - min(p[1] for p in box)
                digit_boxes.append((text, score, height))

        if not digit_boxes:
            return None, 0

        digit_boxes.sort(key=lambda x: x[2], reverse=True)
        return digit_boxes[0][0], digit_boxes[0][1]

    def cleanup(self):
        """
        Liberta explicitamente a memória (GPU e CPU) ocupada por:
        • PaddleOCR (modelos Paddle + workers)
        • RealESRGANer (pesos RRDBNet na GPU)
        Chama-a depois de 'finalize_ocr', quando não precisas mais dos modelos.
        """

        # ---------- PaddleOCR ----------
        try:
            if hasattr(self, "paddle_ocr") and self.paddle_ocr is not None:
                # versões < 2.7 guardavam os sub-modelos como atributos públicos
                for attr in ("text_detector", "text_recognizer", "text_classifier"):
                    if hasattr(self.paddle_ocr, attr):
                        setattr(self.paddle_ocr, attr, None)

                # fecha pool multiprocessing (caso esteja ativado)
                if hasattr(self.paddle_ocr, "proc_pool") and self.paddle_ocr.proc_pool:
                    self.paddle_ocr.proc_pool.terminate()
                    self.paddle_ocr.proc_pool.join()
                    self.paddle_ocr.proc_pool = None

                del self.paddle_ocr
                self.paddle_ocr = None
        except Exception as e:
            log.warning(f"[cleanup] PaddleOCR: {e}")

        # ---------- Real-ESRGAN ----------
        try:
            if hasattr(self, "upscaler") and self.upscaler is not None:
                self.upscaler = None
        except Exception as e:
            log.warning(f"[cleanup] RealESRGAN: {e}")

        # ---------- Torch ----------
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # em CPU

        # ---------- Paddle ----------
        try:
            if paddle.device.is_compiled_with_cuda():
                try:
                    paddle.device.cuda.empty_cache()
                except AttributeError:
                    # fallback para versões < 2.5
                    paddle.fluid.core._cuda_empty_cache()
        except Exception:
            pass  # Paddle em CPU ou versão sem CUDA

        # ---------- GC ----------
        gc.collect()
        log.info("[cleanup] Memória de OCR+SR libertada.")
