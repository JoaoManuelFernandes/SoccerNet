"""
Eval team_side accuracy by matching predicted track_ids to GT track_ids via bbox IoU,
then comparing predicted team vs GT team. No jersey number dependency.

Filters:
  - Only GT role == "player"
  - match_ratio >= 0.3  (tracklet matched to GT at least 30% of frames)
  - n_frames >= MIN_FRAMES (ignore ghost/short tracklets)
"""
import json
import pickle
import zipfile
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

MIN_FRAMES = 5     # ignore tracklets shorter than this
IOU_THRESHOLD = 0.3
MATCH_RATIO_MIN = 0.3
PURITY_MIN = 0.8   # min fraction of matched frames going to same GT track (filters fused tracklets)


def iou_ltwh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx);  iy = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix) * max(0, iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def build_gt_tracks(gt_path):
    """Returns (tracks dict, frame_anns dict).
    tracks: gt_track_id -> {team, role, jersey}
    frame_anns: image_id -> list of {track_id, bbox, team, role, jersey}
    """
    gt = json.load(open(gt_path))
    tracks = {}
    frame_anns = defaultdict(list)
    for ann in gt["annotations"]:
        if "attributes" not in ann or "bbox_image" not in ann:
            continue
        attr = ann["attributes"]
        tid = ann["track_id"]
        iid = str(ann["image_id"])
        b = ann["bbox_image"]
        frame_anns[iid].append({
            "track_id": tid,
            "bbox": [b["x"], b["y"], b["w"], b["h"]],
            "team": attr.get("team"),
            "role": attr.get("role"),
            "jersey": attr.get("jersey"),
        })
        if tid not in tracks:
            tracks[tid] = {
                "team": attr.get("team"),
                "role": attr.get("role"),
                "jersey": attr.get("jersey"),
            }
    return tracks, frame_anns


def match_predicted_to_gt(det_df, frame_anns):
    """For each predicted detection, find best-IoU GT annotation in same frame.
    Returns: pred_track_id -> Counter(gt_track_id -> frame_count)
    """
    matches = defaultdict(Counter)
    for _, row in det_df.iterrows():
        iid = str(row["image_id"])
        pred_tid = row["track_id"]
        if pd.isna(pred_tid):
            continue
        pred_bbox = row["bbox_ltwh"]
        if not isinstance(pred_bbox, (list, np.ndarray)) or len(pred_bbox) < 4:
            continue

        best_iou, best_gt = 0.0, None
        for g in frame_anns.get(iid, []):
            iou = iou_ltwh(pred_bbox, g["bbox"])
            if iou > best_iou:
                best_iou, best_gt = iou, g["track_id"]

        if best_iou >= IOU_THRESHOLD and best_gt is not None:
            matches[pred_tid][best_gt] += 1
    return matches


def eval_video(code, pklz_path, gt_base, min_frames=MIN_FRAMES):
    sngs = f"SNGS-{code}"
    gt_path = f"{gt_base}/{sngs}/Labels-GameState.json"

    gt_tracks, frame_anns = build_gt_tracks(gt_path)

    with zipfile.ZipFile(pklz_path) as zf:
        with zf.open(f"{code}.pkl") as f:
            det = pickle.load(f)

    players = det[det.role == "player"].copy()
    matches = match_predicted_to_gt(players, frame_anns)

    results = []
    for pred_tid, gt_counter in matches.items():
        gt_tid = gt_counter.most_common(1)[0][0]
        gt_info = gt_tracks.get(gt_tid, {})
        gt_team = gt_info.get("team")
        gt_role = gt_info.get("role")

        pred_rows = players[players.track_id == pred_tid]
        n_frames = len(pred_rows)
        pred_teams = pred_rows["team"].dropna()
        if len(pred_teams) == 0:
            continue
        pred_team = pred_teams.mode()[0]

        n_matched = gt_counter.most_common(1)[0][1]
        match_ratio = n_matched / n_frames
        total_iou_matches = sum(gt_counter.values())
        purity = n_matched / total_iou_matches if total_iou_matches > 0 else 0.0

        results.append({
            "pred_tid": int(pred_tid),
            "gt_tid": gt_tid,
            "gt_role": gt_role,
            "gt_jersey": gt_info.get("jersey"),
            "purity": round(purity, 2),
            "gt_team": gt_team,
            "pred_team": pred_team,
            "correct": (pred_team == gt_team) if gt_team else None,
            "n_frames": n_frames,
            "match_ratio": round(match_ratio, 2),
        })

    df = pd.DataFrame(results)
    if df.empty:
        print(f"{sngs}: no results")
        return df

    valid = df[
        (df.gt_role == "player") &
        (df.match_ratio >= MATCH_RATIO_MIN) &
        (df.n_frames >= min_frames) &
        (df.purity >= PURITY_MIN) &
        df.correct.notna()
    ].copy()
    valid["correct"] = valid["correct"].astype(bool)

    correct = valid.correct.sum()
    total = len(valid)
    skipped = len(df) - total
    pct = 100 * correct / total if total > 0 else float("nan")
    print(f"{sngs}:  {correct:>3}/{total:<3} = {pct:5.1f}%   (skipped {skipped} short/unmatched)")

    wrong = valid[~valid.correct]
    if len(wrong):
        print("  WRONG:")
        for _, r in wrong.iterrows():
            print(f"    pred_tid={r.pred_tid:>4}  gt_tid={r.gt_tid:>4}  "
                  f"jersey={str(r.gt_jersey):>3}  gt={str(r.gt_team):<6} "
                  f"pred={str(r.pred_team):<6}  frames={r.n_frames}")
    return valid


if __name__ == "__main__":
    import sys
    import os

    GT_BASE = "/home/joao/soccernet/data/SoccerNetGS/valid"

    # Accept pklz path as argument, otherwise use latest
    if len(sys.argv) > 1:
        PKLZ = sys.argv[1]
    else:
        PKLZ = "/home/joao/soccernet/outputs/sn-gamestate/2026-05-24/17-39-40/states/sn-gamestate-color-reid.pklz"

    # Discover which video codes are in the pklz
    with zipfile.ZipFile(PKLZ) as zf:
        codes = sorted([
            n.replace(".pkl", "") for n in zf.namelist()
            if n.endswith(".pkl") and not n.endswith("_image.pkl") and n != "summary.json"
        ])

    print(f"Evaluating {len(codes)} videos from: {os.path.basename(PKLZ)}")
    print(f"Min frames filter: {MIN_FRAMES}\n")

    all_dfs = []
    for code in codes:
        df = eval_video(code, PKLZ, GT_BASE)
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs)
        total = len(combined)
        correct = int(combined.correct.sum())
        print(f"\n{'='*50}")
        print(f"GLOBAL:  {correct}/{total} = {100*correct/total:.1f}%")
