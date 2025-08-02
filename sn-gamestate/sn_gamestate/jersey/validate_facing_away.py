import os
import pandas as pd
from sn_gamestate.jersey.mmocr_utils import is_facing_away
import argparse

POSE_METRICS_DIR = "/home/joao/soccernet/outputs/sn-gamestate/2025-06-24/17-39-09/debug_pose_metrics"


def extract_keypoints_and_scores(row):
    # Extrai apenas os scores do CSV
    keypoints = {
        'left_shoulder': (0, 0),
        'right_shoulder': (0, 0),
        'left_hip': (0, 0),
        'right_hip': (0, 0),
        'left_eye': (0, 0),
        'right_eye': (0, 0),
        'nose': (0, 0),
    }
    keypoint_scores = {
        'left_eye': float(row['left_eye_score']),
        'right_eye': float(row['right_eye_score']),
        'nose': float(row['nose_score']),
        'left_shoulder': 1.0,
        'right_shoulder': 1.0,
        'left_hip': 1.0,
        'right_hip': 1.0,
    }
    return keypoints, keypoint_scores


def is_facing_away_with_angle(keypoints, keypoint_scores, angle):
    leye_score = keypoint_scores['left_eye']
    reye_score = keypoint_scores['right_eye']
    nose_score = keypoint_scores['nose']
    low_07 = [leye_score < 0.7, reye_score < 0.7, nose_score < 0.7]
    low_08 = [leye_score < 0.8, reye_score < 0.8, nose_score < 0.8]
    if angle > 175:
        return sum(low_07) >= 2
    elif 150 < angle <= 175:
        return sum(low_08) >= 2
    elif 145 < angle <= 150:
        return sum(low_08) >= 1
    else:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_percent_away', type=float, default=70.0, help='Percentual mínimo de frames de costas para enviar tracklet ao OCR')
    args = parser.parse_args()
    min_percent_away = args.min_percent_away

    results = []
    for fname in sorted(os.listdir(POSE_METRICS_DIR)):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(POSE_METRICS_DIR, fname)
        df = pd.read_csv(fpath)
        n_frames = len(df)
        n_away = 0
        away_frames = []
        for idx, row in df.iterrows():
            keypoints, keypoint_scores = extract_keypoints_and_scores(row)
            angle = float(row['angle'])
            if is_facing_away_with_angle(keypoints, keypoint_scores, angle):
                n_away += 1
                away_frames.append(row['frame_id'])
        percent = 100.0 * n_away / n_frames if n_frames > 0 else 0.0
        track_id = df['track_id'].iloc[0] if n_frames > 0 else fname
        results.append((track_id, n_away, n_frames, percent, away_frames))

    print("TrackID | Frames de costas | Total | % | Frames | Enviar_OCR")
    for track_id, n_away, n_frames, percent, away_frames in results:
        enviar_ocr = percent >= min_percent_away
        print(f"{track_id:7} | {n_away:16} | {n_frames:5} | {percent:5.1f}% | {str(away_frames)[:80]}... | {enviar_ocr}")


if __name__ == "__main__":
    main()
