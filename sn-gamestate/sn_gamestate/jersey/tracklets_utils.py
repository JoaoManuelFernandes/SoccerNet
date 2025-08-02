import os
import cv2
import numpy as np
import pandas as pd
import logging
import math
log = logging.getLogger(__name__)



ANGLE_THRESHOLD = 165
MIN_EYE_SCORE = 0.85
MIN_EAR_SCORE = 0.85
MIN_NOSE_SCORE = 0.85

###############################################################
# CONFIGURACAO: Definir esqueleto COCO (17 keypoints)
###############################################################
COCO_SKELETON = [
    (0, 1),  # nose -> left_eye
    (0, 2),  # nose -> right_eye
    (1, 3),  # left_eye -> left_ear
    (2, 4),  # right_ear -> right_ear
    (5, 6),  # left_shoulder -> right_shoulder
    (5, 7),  # left_shoulder -> left_elbow
    (7, 9),  # left_elbow -> left_wrist
    (6, 8),  # right_shoulder -> right_elbow
    (8, 10), # right_elbow -> right_wrist
    (5, 11), # left_shoulder -> left_hip
    (6, 12), # right_shoulder -> right_hip
    (11, 12),# left_hip -> right_hip
    (11, 13),# left_hip -> left_knee
    (13, 15),# left_knee -> left_ankle
    (12, 14),# right_hip -> right_knee
    (14, 16) # right_knee -> right_ankle
]

def run_hrnet_pose_inference(pose_inferencer, image_rgb):
    """
    Lê 'image_rgb' (H,W,3) e retorna (keypoints_dict, keypoints_scores) ou (None, None), usando MMPose 1.x.
    """
    if pose_inferencer is None:
        return None, None

    image_bgr = image_rgb[..., ::-1]
    results_generator = pose_inferencer(image_bgr, return_vis=False)
    results_list = list(results_generator)
    if not results_list:
        return None, None

    result_obj = results_list[0]
    if 'predictions' not in result_obj:
        return None, None

    predictions_list = result_obj['predictions']
    if not predictions_list:
        return None, None

    persons = predictions_list[0]
    if not persons:
        return None, None

    first_person = persons[0]
    if 'keypoints' not in first_person or 'keypoint_scores' not in first_person:
        return None, None

    kpts = first_person['keypoints']         # (17, 2)
    scores = first_person['keypoint_scores'] # (17,)
    if len(kpts) < 17 or len(scores) < 17:
        return None, None

    # Mapeamento COCO
    coco_kpt_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    keypoints_dict = {}
    keypoints_scores = {}

    for i, name in enumerate(coco_kpt_names):
        x, y = kpts[i]
        score = scores[i]
        keypoints_dict[name] = (float(x), float(y))
        keypoints_scores[name] = float(score)

    return keypoints_dict, keypoints_scores

def angle_between_vectors(v1, v2):
    """
    Retorna o ângulo em graus entre dois vetores 2D (v1 e v2).
    v1 e v2 são tuplas (x, y).
    """
    x1, y1 = v1
    x2, y2 = v2
    
    # Produto escalar
    dot = x1 * x2 + y1 * y2
    
    # Magnitudes
    mag1 = math.sqrt(x1**2 + y1**2)
    mag2 = math.sqrt(x2**2 + y2**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0  # Evita divisões por zero; pode ajustar de acordo com seu caso
    
    # cos(theta) = dot / (mag1 * mag2)
    cos_angle = dot / (mag1 * mag2)
    
    # Para evitar problemas numéricos se cos_angle ligeiramente passar de [-1, 1]
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    
    # Converte para graus
    angle_degrees = math.degrees(math.acos(cos_angle))
    return angle_degrees

def is_facing_away(keypoints, keypoint_scores, angle_threshold_1=175, angle_threshold_2=150, angle_threshold_3=145, score_threshold_1=0.7, score_threshold_2=0.8):
    """
    Retorna True se provavelmente o jogador está de costas, com base em thresholds configuráveis.
    """
    needed = [
        'left_shoulder', 'right_shoulder',
        'left_hip', 'right_hip',
        'left_eye', 'right_eye',
        'nose'
    ]
    for n in needed:
        if n not in keypoints or n not in keypoint_scores:
            return False

    ls = np.array(keypoints['left_shoulder'])
    rs = np.array(keypoints['right_shoulder'])
    lh = np.array(keypoints['left_hip'])
    rh = np.array(keypoints['right_hip'])
    leye = np.array(keypoints['left_eye'])
    reye = np.array(keypoints['right_eye'])

    leye_score = keypoint_scores['left_eye']
    reye_score = keypoint_scores['right_eye']
    nose_score = keypoint_scores['nose']

    eye_center = (leye + reye) / 2
    mid_shoulder = (ls + rs) / 2
    mid_hip = (lh + rh) / 2
    trunk_vector = mid_hip - mid_shoulder
    head_vector = eye_center - mid_shoulder
    angle = angle_between_vectors(trunk_vector, head_vector)

    low_1 = [leye_score < score_threshold_1, reye_score < score_threshold_1, nose_score < score_threshold_1]
    low_2 = [leye_score < score_threshold_2, reye_score < score_threshold_2, nose_score < score_threshold_2]

    if angle > angle_threshold_1:
        return sum(low_1) >= 2
    elif angle_threshold_2 < angle <= angle_threshold_1:
        return sum(low_2) >= 2
    elif angle_threshold_3 < angle <= angle_threshold_2:
        return sum(low_2) >= 1
    else:
        return False

def crop_back_region(image_rgb, keypoints):
    """
    Recorta a região das costas (ombros -> quadris).
    """
    needed = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    for n in needed:
        if n not in keypoints:
            return image_rgb

    ls_x, ls_y = keypoints['left_shoulder']
    rs_x, rs_y = keypoints['right_shoulder']
    lh_x, lh_y = keypoints['left_hip']
    rh_x, rh_y = keypoints['right_hip']

    x_min = int(min(ls_x, rs_x))
    x_max = int(max(ls_x, rs_x))
    y_top = int(min(ls_y, rs_y) - 10)
    y_bottom = int(max(lh_y, rh_y) + 10)

    h, w, _ = image_rgb.shape
    x_min = max(0, x_min)
    x_max = min(w, x_max)
    y_top = max(0, y_top)
    y_bottom = min(h, y_bottom)

    return image_rgb[y_top:y_bottom, x_min:x_max]

def debug_save_image(
    output_path,
    image_rgb,
    keypoints_dict=None,
    skeleton=None,
    bbox_ltwh=None,
    debug_text="",
    color_pose=(0, 255, 0),
    color_skel=(255, 0, 0),
    color_bbox=(0, 255, 255)
):
    """
    Salva 'image_rgb' (np.uint8, RGB) em disco, desenhando keypoints, skeleton, bbox, etc.
    """

    # Proteção contra imagem inválida
    if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        print(f"[WARNING] Imagem inválida recebida. Ignorando gravação: {output_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        debug_img_bgr = image_rgb[..., ::-1].copy()
    except Exception as e:
        print(f"[ERROR] Falha ao converter imagem RGB para BGR: {e}")
        return

    # Desenha bbox + texto
    if bbox_ltwh and len(bbox_ltwh) >= 4:
        l, t, w, h = bbox_ltwh
        r = l + w
        b = t + h
        cv2.rectangle(debug_img_bgr, (l, t), (r, b), color_bbox, 2)
        if debug_text:
            cv2.putText(debug_img_bgr, debug_text, (l, t - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bbox, 1)

    # Desenha keypoints
    if keypoints_dict is not None:
        for (kx, ky) in keypoints_dict.values():
            cv2.circle(debug_img_bgr, (int(kx), int(ky)), 3, color_pose, 2)

    # Desenha skeleton
    if skeleton is not None and keypoints_dict is not None:
        coco_kpt_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        for (i1, i2) in skeleton:
            if i1 < 0 or i1 >= len(coco_kpt_names): 
                continue
            if i2 < 0 or i2 >= len(coco_kpt_names):
                continue

            name1 = coco_kpt_names[i1]
            name2 = coco_kpt_names[i2]
            if name1 not in keypoints_dict or name2 not in keypoints_dict:
                continue

            x1, y1 = keypoints_dict[name1]
            x2, y2 = keypoints_dict[name2]
            cv2.line(debug_img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color_skel, 2)

    #try:
    #    cv2.imwrite(output_path, debug_img_bgr)
    #    print(f"[DEBUG] Salvou imagem com anotações em: {output_path}")
    #except Exception as e:
    #    print(f"[ERROR] Falha ao salvar imagem em {output_path}: {e}")

###############################################################
# 4) Funções OCR genérico e salvamento
###############################################################
# As funções save_ocr_results e alternative_jersey_number_detection foram removidas pois não são utilizadas no pipeline.

def save_images_by_tracklet(detections, images_np, metadatas, tracklet_images, output_dir):
    #if not os.path.exists(output_dir):
        #os.makedirs(output_dir)

    idx = 0
    for _, detection in detections.iterrows():
        track_id = detection.get('track_id', None)
        frame_id = detection.get('image_id', None)

        if track_id is None or frame_id is None:
            continue
        if idx >= len(images_np):
            continue

        image_rgb = images_np[idx]
        if image_rgb is None or image_rgb.size == 0:
            continue

        # Armazena no dict
        if track_id not in tracklet_images:
            tracklet_images[track_id] = []
        tracklet_images[track_id].append((frame_id, image_rgb))

        # Salva em disco
        #tracklet_dir = os.path.join(output_dir, f'tracklet_{track_id}')
        #os.makedirs(tracklet_dir, exist_ok=True)

        #image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        #filename = os.path.join(tracklet_dir, f'frame_{frame_id}.jpg')
        #cv2.imwrite(filename, image_bgr) 

        idx += 1

    return tracklet_images

def compute_pose_metrics(keypoints, keypoint_scores):
    """Calcula métricas úteis de pose para debug usando limiares globais."""
    metrics = {}
    try:
        # Extrai coordenadas
        ls = np.array(keypoints.get('left_shoulder', [0, 0]))
        rs = np.array(keypoints.get('right_shoulder', [0, 0]))
        lh = np.array(keypoints.get('left_hip', [0, 0]))
        rh = np.array(keypoints.get('right_hip', [0, 0]))
        leye = np.array(keypoints.get('left_eye', [0, 0]))
        reye = np.array(keypoints.get('right_eye', [0, 0]))
        nose = np.array(keypoints.get('nose', [0, 0]))

        # Extrai scores
        lear_score = keypoint_scores.get('left_ear', 0.0)
        rear_score = keypoint_scores.get('right_ear', 0.0)
        leye_score = keypoint_scores.get('left_eye', 0.0)
        reye_score = keypoint_scores.get('right_eye', 0.0)
        nose_score = keypoint_scores.get('nose', 0.0)

        # Cálculo de ângulo tronco–cabeça
        mid_shoulder = (ls + rs) / 2
        mid_hip = (lh + rh) / 2
        eye_center = (leye + reye) / 2
        angle = angle_between_vectors(mid_hip - mid_shoulder, eye_center - mid_shoulder)

        # Flags individuais usando limiares globais
        shoulder_inversion = float(ls[0] > rs[0])
        low_eye = float(leye_score < MIN_EYE_SCORE or reye_score < MIN_EYE_SCORE)
        low_ear = float(lear_score < MIN_EAR_SCORE or rear_score < MIN_EAR_SCORE)
        low_nose = float(nose_score < MIN_NOSE_SCORE)

        # Usa defaults de is_facing_away (que referenciam as mesmas constantes)
        facing_away_flag = float(is_facing_away(keypoints, keypoint_scores))

        # Atualiza dicionário de métricas
        metrics.update({
            'angle': float(angle),
            'left_ear_score': lear_score,
            'right_ear_score': rear_score,
            'left_eye_score': leye_score,
            'right_eye_score': reye_score,
            'nose_score': nose_score,
            'shoulder_inversion': shoulder_inversion,
            'low_eye': low_eye,
            'low_ear': low_ear,
            'low_nose': low_nose,
            'facing_away': facing_away_flag
        })
    except Exception as e:
        log.warning(f"Erro ao computar métricas de pose: {e}")
    return metrics


def save_pose_debug_data(pose_debug_data, run_path, prefix="track_pose_metrics"):
    """Salva dicionário pose_debug_data {track_id: [ {frame_id, ...}, ... ]} em CSVs."""
    if not pose_debug_data:
        return
    pose_debug_dir = os.path.join(run_path, "debug_pose_metrics")
    os.makedirs(pose_debug_dir, exist_ok=True)
    for track_id, entries in pose_debug_data.items():
        try:
            df_pose = pd.DataFrame(entries)
            out_file = os.path.join(pose_debug_dir, f"{prefix}_track_{track_id}.csv")
            df_pose.to_csv(out_file, index=False)
            log.info(f"Pose metrics salvo: {out_file}")
        except Exception as e:
            log.warning(f"Falha ao salvar pose metrics para track {track_id}: {e}")

def is_facing_away_trunk_motion(keypoints, keypoint_scores, motion_vec=None, angle_trunk_motion_threshold=120, angle_threshold_1=175, angle_threshold_2=150, angle_threshold_3=145, score_threshold_1=0.7, score_threshold_2=0.8):
    """
    Heurística combinada: só retorna True se heurística tradicional + orientação do tronco oposta ao movimento.
    Se motion_vec for None ou muito pequeno, loga e usa apenas heurística tradicional.
    """
    base_away = is_facing_away(
        keypoints, keypoint_scores,
        angle_threshold_1=angle_threshold_1,
        angle_threshold_2=angle_threshold_2,
        angle_threshold_3=angle_threshold_3,
        score_threshold_1=score_threshold_1,
        score_threshold_2=score_threshold_2
    )
    if motion_vec is None or np.linalg.norm(motion_vec) < 1e-3:
        # Log para debug
        #print('[DEBUG] motion_vec muito pequeno ou None, usando apenas heurística tradicional')
        return base_away
    if not base_away:
        return False
    ls = np.array(keypoints['left_shoulder'])
    rs = np.array(keypoints['right_shoulder'])
    lh = np.array(keypoints['left_hip'])
    rh = np.array(keypoints['right_hip'])
    mid_shoulder = (ls + rs) / 2
    mid_hip = (lh + rh) / 2
    trunk_vec = mid_hip - mid_shoulder
    trunk_norm = np.linalg.norm(trunk_vec)
    if trunk_norm < 1e-3:
        #print('[DEBUG] trunk_vec muito pequeno, usando apenas heurística tradicional')
        return base_away
    trunk_unit = trunk_vec / trunk_norm
    dot = np.dot(trunk_unit, motion_vec)
    dot = np.clip(dot, -1.0, 1.0)
    angle_trunk_motion = float(np.degrees(np.arccos(dot)))
    logging.getLogger(__name__).info(f'[TRUNK_MOTION_LOG] angle_trunk_motion={angle_trunk_motion:.2f} (threshold={angle_trunk_motion_threshold})')
    return angle_trunk_motion > angle_trunk_motion_threshold
