import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import math
log = logging.getLogger(__name__)



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


def is_facing_away(
    keypoints,
    keypoint_scores,
    angle_threshold=165,
    min_eye_score=0.6,
    min_ear_score=0.5,
    min_nose_score=0.4
):
    needed = [
        'left_shoulder', 'right_shoulder',
        'left_hip', 'right_hip',
        'left_eye', 'right_eye',
        'left_ear', 'right_ear',
        'nose', 'left_wrist', 'right_wrist'
    ]
    for n in needed:
        if n not in keypoints or n not in keypoint_scores:
            print(f"[DEBUG] Keypoint ausente: {n}")
            return False

    # Posições
    ls = np.array(keypoints['left_shoulder'])
    rs = np.array(keypoints['right_shoulder'])
    lh = np.array(keypoints['left_hip'])
    rh = np.array(keypoints['right_hip'])
    leye = np.array(keypoints['left_eye'])
    reye = np.array(keypoints['right_eye'])
    nose = np.array(keypoints['nose'])
    lwrist = np.array(keypoints['left_wrist'])
    rwrist = np.array(keypoints['right_wrist'])

    # Scores
    lear = keypoint_scores['left_ear']
    rear = keypoint_scores['right_ear']
    leye_score = keypoint_scores['left_eye']
    reye_score = keypoint_scores['right_eye']
    nose_score = keypoint_scores['nose']

    score = 0

    # 1) Confiança baixa em olhos e nariz
    if leye_score < min_eye_score or reye_score < min_eye_score:
        print("[DEBUG] Olhos com baixa confiança")
        score += 1
    if min_nose_score < nose_score:
        print("[DEBUG] Nariz com baixa confiança")
        score += 1

    eye_center = (leye + reye) / 2
    # 3) Ângulo entre cabeça e tronco
    mid_shoulder = (ls + rs) / 2
    mid_hip = (lh + rh) / 2
    trunk_vector = mid_hip - mid_shoulder
    head_vector = eye_center - mid_shoulder
    angle = angle_between_vectors(trunk_vector, head_vector)
    print(f"[DEBUG] Ângulo tronco-cabeça: {angle:.1f}°")
    if angle > angle_threshold:
        print("[DEBUG] Ângulo grande → costas")
        score += 1

    # 4) Ombros invertidos
    if rs[0] > ls[0]:
        print("[DEBUG] Ombros invertidos (esq > dir)")
        score += 1

    # Decisão
    print(f"[DEBUG] Score total: {score}")
    return score >= 3



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

    try:
        cv2.imwrite(output_path, debug_img_bgr)
        print(f"[DEBUG] Salvou imagem com anotações em: {output_path}")
    except Exception as e:
        print(f"[ERROR] Falha ao salvar imagem em {output_path}: {e}")

###############################################################
# 4) Funções OCR genérico e salvamento
###############################################################
def save_ocr_results(detections: pd.DataFrame, output_file="ocr_results_direct.csv"):
    required = [
        "track_id", "image_id",
        "jersey_number_detection", "jersey_number_confidence"
    ]
    missing = [col for col in required if col not in detections.columns]

    if missing:
        log.warning(f"⚠️ Colunas ausentes no DataFrame: {missing}. Nenhum arquivo foi salvo.")
        return

    output_df = pd.DataFrame({
        "track_id": detections["track_id"],
        "frame_id": detections["image_id"],
        "ocr_direct_jn": detections["jersey_number_detection"],
        "ocr_direct_conf": detections["jersey_number_confidence"]
    })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Verifica se o ficheiro já existe
    file_exists = os.path.isfile(output_file)

    # Grava em append, só adiciona o header se o ficheiro ainda não existe
    output_df.to_csv(
        output_file,
        mode='a',
        index=False,
        header=not file_exists
    )

    print(f"✅ {len(output_df)} resultados do OCR direto adicionados a: {output_file}")

def alternative_jersey_number_detection(images_np, save_images=True, output_dir="debug_images"):
    if save_images:
        os.makedirs(output_dir, exist_ok=True)

    jersey_numbers = []
    confidences = []

    for i, img_rgb in enumerate(images_np):
        simulated_number = str(np.random.randint(0, 99))
        simulated_confidence = np.random.uniform(0.5, 1.0)

        jersey_numbers.append(simulated_number)
        confidences.append(simulated_confidence)

        if save_images:
            filename = os.path.join(output_dir, f"player_{i}_number_{simulated_number}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            print(f"✅ Imagem salva: {filename}")

    return jersey_numbers, confidences

def save_images_by_tracklet(detections, images_np, metadatas, tracklet_images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        """ tracklet_dir = os.path.join(output_dir, f'tracklet_{track_id}')
        os.makedirs(tracklet_dir, exist_ok=True)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        filename = os.path.join(tracklet_dir, f'frame_{frame_id}.jpg')
        cv2.imwrite(filename, image_bgr) """

        idx += 1

    return tracklet_images
