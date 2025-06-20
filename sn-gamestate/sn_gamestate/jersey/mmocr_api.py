import pandas as pd
import torch
import numpy as np
import os
import math
import logging
import cv2
import time

import re
from collections import defaultdict, Counter
import gc, paddle, torch
print("paddle a usar:" + paddle.device.get_device())        # deve devolver 'gpu:0'

from paddleocr import PaddleOCR
from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.utils import bbox2poly, crop_img, poly2bbox

from mmengine.registry import init_default_scope
init_default_scope('mmpose')

from mmpose.apis import MMPoseInferencer

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)

from sn_gamestate.jersey.mmocr_utils import (
    run_hrnet_pose_inference,
    is_facing_away,
    crop_back_region,
    debug_save_image,
    save_images_by_tracklet,
    COCO_SKELETON,
    compute_pose_metrics,
    save_pose_debug_data
)


log = logging.getLogger(__name__)

class MMOCR(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, batch_size, device, **kwargs):
        super().__init__(batch_size=batch_size)
        self.device = device
        self.images_debugJNR = kwargs.get("debugJNR", True)
        self.confidence_threshold = kwargs.get("confidenceOCR", 0.75)
        self.batch_size = batch_size
        self.jnr_batch_frames = kwargs.get("jnr_batch_frames", 5)
        self.use_superres = kwargs.get("use_superres", True)
        self.pre_proc = kwargs.get("pre_proc", True) #piores resultados com pre_proc=True

        self.pose_debug_data = defaultdict(list)

        self.best_by_track = {}

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

        # 1) Preparar colunas iniciais
        images_np = [img.cpu().numpy() for img in batch['img']]
        del batch['img']
        detections['jersey_number_detection'] = None
        detections['jersey_number_confidence'] = 0.0
        detections['ocr_tracklet_pose_jn']   = np.nan
        detections['ocr_tracklet_pose_conf'] = 0.0

        # 2) Acumular imagens por tracklet
        tracklets_output_dir = os.path.join(run_path, "tracklets_images")
        save_images_by_tracklet(
            detections=detections,
            images_np=images_np,
            metadatas=metadatas,
            tracklet_images=self.tracklet_images_global,
            output_dir=tracklets_output_dir
        )

        # 3) Para cada tracklet pronto, rodar pose+OCR
        for track_id, imgs_list in list(self.tracklet_images_global.items()):
            if len(imgs_list) < self.jnr_batch_frames:
                continue

            cropped_list = []
            for frame_id, raw_img in imgs_list:
                if raw_img is None or raw_img.size == 0 or self.pose_model is None:
                    #cropped_list.append((frame_id, raw_img))
                    continue

                # Inferência de pose
                keypoints, keypoints_scores = run_hrnet_pose_inference(self.pose_model, raw_img)
                if keypoints is None or keypoints_scores is None:
                    #cropped_list.append((frame_id, raw_img))
                    continue

                # Computar métricas de pose e armazenar em self.pose_debug_data
                try:
                    from sn_gamestate.jersey.mmocr_utils import compute_pose_metrics
                    metrics = compute_pose_metrics(keypoints, keypoints_scores)
                    entry = {"track_id": track_id, "frame_id": frame_id}
                    entry.update(metrics)
                    self.pose_debug_data[track_id].append(entry)
                except Exception as e:
                    log.warning(f"[process] Falha ao computar métricas de pose para track {track_id}, frame {frame_id}: {e}")

                # Decidir crop de costas ou manter inteiro
                if is_facing_away(keypoints, keypoints_scores):
                    back_img = crop_back_region(raw_img, keypoints)
                    # Debug visual
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
                    #cropped_list.append((frame_id, raw_img))

            # Agrupar para OCR
            cropped_list_rgb = [(track_id, fid, img) for fid, img in cropped_list]
            # Rodar OCR por tracklet
            if self.images_debugJNR:
                print(f"run_paddleocr_batch (track_id={track_id}, {len(cropped_list_rgb)} frames)...")
                t0_paddle = time.perf_counter()
                result_dict = self.run_paddleocr_batch(cropped_list_rgb)
                t1_paddle = time.perf_counter()
                print(f"[Timing] Track {track_id}: run_paddleocr_batch levou {t1_paddle - t0_paddle:.2f}s")
            else:
                result_dict = self.run_paddleocr_batch(cropped_list_rgb)

            batch_jn, batch_conf = result_dict.get(track_id, (None, 0.0))
            if batch_jn is not None and self.images_debugJNR:
                print(f"[TrackletOCR] Track {track_id} → jersey {batch_jn} (conf={batch_conf:.2f})")

            frames_do_batch = [fid for (fid, _) in cropped_list]
            if batch_jn is not None:
                prev_jn, prev_conf = self.best_by_track.get(track_id, (None, 0.0))
                if self.should_replace(prev_jn, prev_conf, batch_jn, batch_conf):
                    self.best_by_track[track_id] = (batch_jn, batch_conf)
                    chosen_jn, chosen_conf = batch_jn, batch_conf
                else:
                    chosen_jn, chosen_conf = prev_jn, prev_conf
                mask = detections['image_id'].isin(frames_do_batch)
                detections.loc[mask, 'ocr_tracklet_pose_jn']   = chosen_jn
                detections.loc[mask, 'ocr_tracklet_pose_conf'] = chosen_conf

            # Manter apenas último no buffer
            if imgs_list:
                self.tracklet_images_global[track_id] = [imgs_list[-1]]
            else:
                self.tracklet_images_global[track_id] = []

        # 4) Acumular comparação OCR direto vs tracklet+pose
        for i in range(len(detections)):
            self.comparison_rows.append({
                "track_id": detections.iloc[i].get("track_id"),
                "frame_id": detections.iloc[i].get("image_id"),
                "ocr_direct_jn": detections.iloc[i].get("jersey_number_detection"),
                "ocr_direct_conf": detections.iloc[i].get("jersey_number_confidence"),
                "ocr_tracklet_pose_jn": detections.iloc[i].get("ocr_tracklet_pose_jn"),
                "ocr_tracklet_pose_conf": detections.iloc[i].get("ocr_tracklet_pose_conf")
            })

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
        if self.images_debugJNR:
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

        if self.images_debugJNR:
            cv2.imwrite(os.path.join(ddir, f"{identifier}_out.png"), sr_bgr)

        return sr_bgr

    @torch.no_grad()
    def run_paddleocr_batch(self, track_images):
        if self.images_debugJNR:
            print(">>> ENTER run_paddleocr_batch")
        grouped = defaultdict(list)
        for track_id, frame_id, img in track_images:
            if img is not None and img.size:
                grouped[track_id].append((frame_id, img))

        results_by_track = {}

        for track_id, frames_list in grouped.items():
            if self.images_debugJNR:
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
                if self.images_debugJNR:
                    print(f"    [Timing] SR frame {fid} levou {time.perf_counter()-t0:.3f}s")


            #for i, img_bgr in enumerate(enh_imgs_bgr):
                #fn = os.path.join(debug_dir, f"track{track_id}_crop_{i}.png")
                #cv2.imwrite(fn, img_bgr)
                #print(f"    [Debug] gravou crop em {fn}")

            # --- OCR
            ocr_results = []
            ocr_start = time.perf_counter()
            for idx, img_bgr in enumerate(enh_imgs_bgr):
                if self.pre_proc:
                    img_pre = self.preprocess_for_digits_gray(img_bgr)
                else:
                    img_pre =  cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # dump para debug
                if self.images_debugJNR:
                    debug_dir = os.path.join(os.getenv("HYDRA_RUN_DIR","."), "debug_before_paddle")
                    fn = os.path.join(debug_dir, f"track{track_id}_crop_{idx}.png")
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(fn, img_pre)
                res = self.paddle_ocr.ocr(img_pre, det=True, rec=True, cls=False)
                ocr_results.append(res)
            ocr_time = time.perf_counter() - ocr_start
            if self.images_debugJNR:
                print(f"    [Timing] OCR {len(enh_imgs_bgr)} imgs levou {ocr_time:.3f}s")

            # extrai números
            for fid, res in zip(fid_order, ocr_results):
                jn, conf = self.extract_jersey_numbers_from_paddleocr(res)
                #print(f"    [OCR] frame {fid} -> ({jn}, {conf:.2f})")
                if jn and conf >= self.confidence_threshold:
                    detections.append((jn, conf))

            # votação
            if detections:
                counts = Counter(j for j, _ in detections)
                valid = [j for j,c in counts.items() if c>=min_votes]
                if self.images_debugJNR:
                    print(f"    [Vote] counts={counts}, min_votes={min_votes}, valid={valid}")
                if valid:
                    confs = {j:[c for j2,c in detections if j2==j] for j in valid}
                    best_j = max(confs, key=lambda j: sum(confs[j])/len(confs[j]))
                    best_c = sum(confs[best_j])/len(confs[best_j])
                    results_by_track[track_id] = (best_j, best_c)

                    if self.images_debugJNR:
                        print(f"    [Result] Track {track_id} -> ({best_j}, {best_c:.2f})")
            else:
                if self.images_debugJNR:
                    print(f"    [Result] Track {track_id} -> no detections")
        if self.images_debugJNR:
            print(">>> EXIT run_paddleocr_batch with results:", results_by_track)

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
            if self.should_replace(prev_jn, prev_conf, final_jn, final_conf):
                self.best_by_track[track_id] = (final_jn, final_conf)
            # Além disso, se você coletou métricas de pose em process(), nada a fazer aqui para pose

        # 2) Zera colunas de saída e preenche com o melhor jersey por track
        detections['jersey_number_detection']  = None
        detections['jersey_number_confidence'] = 0.0

        for track_id, (best_jn, best_conf) in self.best_by_track.items():
            if best_jn is None:
                continue
            mask = detections['track_id'] == track_id
            detections.loc[mask, 'jersey_number_detection']  = best_jn
            detections.loc[mask, 'jersey_number_confidence'] = best_conf

        # 3) (Opcional) Limpa o buffer de tracklets
        self.tracklet_images_global.clear()

        # 4) Exporta CSV “por track” baseado em best_by_track
        rows = [
            {"track_id": tid,
            "jersey_number_detection": jn,
            "jersey_number_confidence": conf}
            for tid, (jn, conf) in sorted(self.best_by_track.items())
            if jn is not None
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
        try:
            from sn_gamestate.jersey.mmocr_utils import save_pose_debug_data
            # salva CSVs em run_path/debug_pose_metrics/
            save_pose_debug_data(self.pose_debug_data, run_path, prefix="pose_metrics")
            # limpa buffer de pose
            self.pose_debug_data.clear()
        except Exception as e:
            log.warning(f"[finalize_ocr] Falha ao salvar métricas de pose: {e}")

        # 8) Por fim, limpa best_by_track e debug data de OCR
        self.best_by_track.clear()
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
        if self.images_debugJNR:
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
    
    def should_replace(self, prev_jn, prev_conf, new_jn, new_conf):
        # Se não havia nenhum antes, aceita se new_jn não for None
        if prev_jn is None:
            return new_jn is not None
        if new_jn is None:
            return False
        # conta dígitos
        len_prev = len(prev_jn)
        len_new  = len(new_jn)
        # se o anterior tem 2 dígitos e o novo só 1, não substituir
        if len_prev == 2 and len_new == 1:
            return False
        # se o anterior tem 1 e o novo tem 2, substituir
        if len_prev == 1 and len_new == 2:
            return True
        # se mesmos dígitos, decide por confiança
        return new_conf > prev_conf

    def cleanup(self):
        """
        Liberta explicitamente a memória (GPU e CPU) ocupada por:
        • PaddleOCR (modelos Paddle + workers)
        • RealESRGANer (pesos RRDBNet na GPU)
        Chama-a depois de 'finalize_ocr', quando não precisas mais dos modelos.
        """

        # ---------- PaddleOCR ----------
        try:
            if getattr(self, "paddle_ocr", None) is not None:
                # libertar pesos (VRAM) sem matar workers
                for attr in ("text_detector", "text_recognizer", "text_classifier"):
                    if hasattr(self.paddle_ocr, attr):
                        setattr(self.paddle_ocr, attr, None)
                # NÃO fazer: del self.paddle_ocr  nem mexer em proc_pool
                # o GC cuidará quando o processo acabar
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

    def preprocess_for_digits_gray(self, bgr: np.ndarray) -> np.ndarray:
        """Pré-processamento simplificado e otimizado para memória"""
        # 1) converte pra cinza
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2) realce de contraste local (CLAHE) com parâmetros mais suaves
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(gray)

        # 3) binarização adaptativa (mais eficiente que Otsu para nosso caso)
        binary = cv2.adaptiveThreshold(
            eq,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        # 4) operações morfológicas básicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Converte de volta para BGR (necessário para o OCR)
        return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)