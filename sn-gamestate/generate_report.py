"""
generate_report.py  —  Análise automática pós-pipeline: JNR + Team/Team-side

Uso:
    python generate_report.py <path_to_state.pklz> [--gt_base <path>] [--out <report.md>]

Produz um ficheiro .md com:
  - JNR accuracy por vídeo (via IoU matching ao GT)
  - Team/Team-side accuracy por vídeo (via IoU matching ao GT, purity filter)
  - Diagnóstico de ID switches que afectam ambas as tasks
"""
import argparse
import json
import os
import pickle
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────
IOU_THRESHOLD   = 0.30
MATCH_RATIO_MIN = 0.30
MIN_FRAMES      = 5
PURITY_MIN      = 0.80

GT_BASE_VALID = "/home/joao/soccernet/data/SoccerNetGS/valid"
GT_BASE_TEST  = "/home/joao/soccernet/data/SoccerNetGS/test"


# ── helpers ───────────────────────────────────────────────────────────────────
def iou_ltwh(a, b):
    ax, ay, aw, ah = a;  bx, by, bw, bh = b
    inter = max(0, min(ax+aw, bx+bw) - max(ax, bx)) * \
            max(0, min(ay+ah, by+bh) - max(ay, by))
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def load_gt(gt_path):
    gt = json.load(open(gt_path))
    tracks = {}
    frame_anns = defaultdict(list)
    for ann in gt["annotations"]:
        if "attributes" not in ann or "bbox_image" not in ann:
            continue
        attr = ann["attributes"]
        tid  = ann["track_id"]
        iid  = str(ann["image_id"])
        b    = ann["bbox_image"]
        frame_anns[iid].append({
            "track_id": tid,
            "bbox":     [b["x"], b["y"], b["w"], b["h"]],
            "team":     attr.get("team"),
            "role":     attr.get("role"),
            "jersey":   attr.get("jersey"),
        })
        if tid not in tracks:
            tracks[tid] = {"team": attr.get("team"),
                           "role": attr.get("role"),
                           "jersey": attr.get("jersey")}
    return tracks, frame_anns


def match_to_gt(det_df, frame_anns):
    """Returns pred_track_id -> Counter(gt_track_id -> n_frames)."""
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


def analyse_video(code, det, gt_base):
    """Returns dicts: jnr_results, team_results, id_switches."""
    sngs = f"SNGS-{code}"
    gt_path = Path(gt_base) / sngs / "Labels-GameState.json"
    if not gt_path.exists():
        return None, None, None

    gt_tracks, frame_anns = load_gt(str(gt_path))

    players = det[det.role == "player"].copy()
    matches  = match_to_gt(players, frame_anns)

    jnr_rows   = []
    team_rows  = []
    id_switch_rows = []

    for pred_tid, gt_counter in matches.items():
        top_gt_tid, top_cnt = gt_counter.most_common(1)[0]
        gt_info   = gt_tracks.get(top_gt_tid, {})
        gt_team   = gt_info.get("team")
        gt_role   = gt_info.get("role")
        gt_jersey = gt_info.get("jersey")

        pred_rows  = players[players.track_id == pred_tid]
        n_frames   = len(pred_rows)
        match_ratio = top_cnt / n_frames
        total_iou   = sum(gt_counter.values())
        purity      = top_cnt / total_iou if total_iou > 0 else 0.0

        # flag ID switch
        is_id_switch = purity < PURITY_MIN and n_frames >= MIN_FRAMES

        # predicted values (mode)
        pred_team   = pred_rows["team"].dropna().mode()
        pred_team   = pred_team[0] if len(pred_team) else None
        pred_jersey = pred_rows["jersey_number"].dropna().mode() \
                      if "jersey_number" in pred_rows.columns else pd.Series([])
        pred_jersey = str(int(pred_jersey[0])) if len(pred_jersey) else None

        base = {
            "pred_tid":    int(pred_tid),
            "gt_tid":      top_gt_tid,
            "gt_role":     gt_role,
            "n_frames":    n_frames,
            "match_ratio": round(match_ratio, 2),
            "purity":      round(purity, 2),
            "is_id_switch": is_id_switch,
        }

        # ── JNR ──
        jnr_row = {**base,
                   "gt_jersey":   gt_jersey,
                   "pred_jersey": pred_jersey}
        if (gt_role == "player" and match_ratio >= MATCH_RATIO_MIN
                and n_frames >= MIN_FRAMES and gt_jersey is not None):
            if is_id_switch:
                jnr_row["eval_status"] = "id_switch"
            elif pred_jersey is None:
                jnr_row["eval_status"] = "no_prediction"
            elif pred_jersey == str(gt_jersey):
                jnr_row["eval_status"] = "correct"
            else:
                jnr_row["eval_status"] = "wrong"
        else:
            jnr_row["eval_status"] = "skipped"
        jnr_rows.append(jnr_row)

        # ── Team ──
        team_row = {**base,
                    "gt_team":   gt_team,
                    "pred_team": pred_team}
        if (gt_role == "player" and match_ratio >= MATCH_RATIO_MIN
                and n_frames >= MIN_FRAMES and purity >= PURITY_MIN
                and gt_team is not None):
            team_row["eval_status"] = "correct" if pred_team == gt_team else "wrong"
        elif is_id_switch and gt_role == "player" and match_ratio >= MATCH_RATIO_MIN:
            team_row["eval_status"] = "id_switch"
        else:
            team_row["eval_status"] = "skipped"
        team_rows.append(team_row)

        # ── ID switch log ──
        if is_id_switch:
            # find the other GT track
            others = [(gt_id, cnt) for gt_id, cnt in gt_counter.most_common()
                      if gt_id != top_gt_tid]
            other_info = []
            for gt_id, cnt in others[:2]:
                info = gt_tracks.get(gt_id, {})
                other_info.append(f"gt_tid={gt_id} jersey={info.get('jersey')} "
                                  f"team={info.get('team')} ({cnt}fr)")
            id_switch_rows.append({
                "pred_tid":   int(pred_tid),
                "n_frames":   n_frames,
                "purity":     round(purity, 2),
                "primary":    f"gt_tid={top_gt_tid} jersey={gt_jersey} team={gt_team} ({top_cnt}fr)",
                "others":     " | ".join(other_info),
                "pred_team":  pred_team,
                "pred_jersey": pred_jersey,
            })

    return pd.DataFrame(jnr_rows), pd.DataFrame(team_rows), pd.DataFrame(id_switch_rows)


# ── report builder ─────────────────────────────────────────────────────────────
def build_report(pklz_path, gt_base, out_path):
    lines = []
    def w(s=""): lines.append(s)

    w(f"# Pipeline Report — JNR & Team/Team-side")
    w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**State file**: `{pklz_path}`")
    w()

    with zipfile.ZipFile(pklz_path) as zf:
        codes = sorted([
            n.replace(".pkl", "") for n in zf.namelist()
            if n.endswith(".pkl") and not n.endswith("_image.pkl")
            and n != "summary.json"
        ])
        all_dets = {}
        for code in codes:
            with zf.open(f"{code}.pkl") as f:
                all_dets[code] = pickle.load(f)

    w(f"**Videos**: {', '.join(f'SNGS-{c}' for c in codes)}")
    w(f"**Eval filters**: min_frames={MIN_FRAMES}, match_ratio≥{MATCH_RATIO_MIN}, "
      f"iou≥{IOU_THRESHOLD}, purity≥{PURITY_MIN} (team only)")
    w()

    all_jnr, all_team, all_switches = [], [], []

    # ── Per-video ──────────────────────────────────────────────────────────────
    w("---")
    w("## Per-video Results")
    w()

    for code in codes:
        sngs = f"SNGS-{code}"
        det  = all_dets[code]
        jnr_df, team_df, sw_df = analyse_video(code, det, gt_base)

        if jnr_df is None:
            w(f"### {sngs}  ⚠️ GT not found"); w(); continue

        w(f"### {sngs}")

        # pipeline columns present?
        has_jnr  = "jersey_number" in det.columns and det["jersey_number"].notna().any()
        has_team = "team" in det.columns and det["team"].notna().any()
        w(f"- JNR column present: {'✓' if has_jnr else '✗'}  |  "
          f"Team column present: {'✓' if has_team else '✗'}")
        w()

        # ── JNR table ──
        w("#### JNR")
        jnr_eval = jnr_df[jnr_df.eval_status != "skipped"]
        jnr_clean = jnr_eval[jnr_eval.eval_status != "id_switch"]
        jnr_correct  = (jnr_clean.eval_status == "correct").sum()
        jnr_wrong    = (jnr_clean.eval_status == "wrong").sum()
        jnr_nopred   = (jnr_clean.eval_status == "no_prediction").sum()
        jnr_switch   = (jnr_eval.eval_status == "id_switch").sum()
        jnr_total    = len(jnr_clean)
        jnr_pct      = 100*jnr_correct/jnr_total if jnr_total > 0 else float("nan")

        w(f"| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |")
        w(f"|---------|-------|---------------|-------------------|-------------|-------------|")
        w(f"| {jnr_correct} | {jnr_wrong} | {jnr_nopred} | {jnr_switch} | {jnr_total} | **{jnr_pct:.1f}%** |")
        w()

        if jnr_wrong > 0:
            w("Wrong JNR predictions:")
            wrongs = jnr_clean[jnr_clean.eval_status == "wrong"][
                ["pred_tid","gt_jersey","pred_jersey","n_frames","purity"]].head(10)
            w("```")
            w(wrongs.to_string(index=False))
            w("```")
            w()

        # ── Team table ──
        w("#### Team / Team-side")
        team_eval  = team_df[team_df.eval_status != "skipped"]
        team_clean = team_eval[team_eval.eval_status != "id_switch"]
        t_correct  = (team_clean.eval_status == "correct").sum()
        t_wrong    = (team_clean.eval_status == "wrong").sum()
        t_switch   = (team_eval.eval_status == "id_switch").sum()
        t_total    = len(team_clean)
        t_pct      = 100*t_correct/t_total if t_total > 0 else float("nan")

        w(f"| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |")
        w(f"|---------|-------|-------------------|-------------|-------------|")
        w(f"| {t_correct} | {t_wrong} | {t_switch} | {t_total} | **{t_pct:.1f}%** |")
        w()

        if t_wrong > 0:
            w("Wrong team predictions:")
            wrongs = team_clean[team_clean.eval_status == "wrong"][
                ["pred_tid","gt_team","pred_team","n_frames","purity"]].head(10)
            w("```")
            w(wrongs.to_string(index=False))
            w("```")
            w()

        # ── ID switches ──
        if sw_df is not None and len(sw_df) > 0:
            w(f"#### ID Switches detected: {len(sw_df)}")
            w(f"*(purity < {PURITY_MIN} — tracklets tracking 2+ GT players; excluded from JNR+team eval)*")
            w()
            w("```")
            for _, r in sw_df.iterrows():
                w(f"  pred_tid={r.pred_tid:>4}  frames={r.n_frames:>4}  purity={r.purity:.2f}")
                w(f"    primary : {r.primary}")
                w(f"    others  : {r.others}")
                w(f"    predicted: jersey={r.pred_jersey}  team={r.pred_team}")
            w("```")
            w()

        all_jnr.append(jnr_df)
        all_team.append(team_df)
        if sw_df is not None and len(sw_df) > 0:
            sw_df["video"] = sngs
            all_switches.append(sw_df)

    # ── Global summary ─────────────────────────────────────────────────────────
    w("---")
    w("## Global Summary")
    w()

    if all_jnr:
        combined_jnr = pd.concat(all_jnr)
        jnr_eval  = combined_jnr[combined_jnr.eval_status != "skipped"]
        jnr_clean = jnr_eval[jnr_eval.eval_status != "id_switch"]
        jnr_c = (jnr_clean.eval_status == "correct").sum()
        jnr_n = len(jnr_clean)
        jnr_sw = (jnr_eval.eval_status == "id_switch").sum()
        jnr_np = (jnr_clean.eval_status == "no_prediction").sum()
        w(f"### JNR")
        w(f"| Correct | Wrong | No prediction | ID-switch (excl.) | Total clean | **Accuracy** |")
        w(f"|---------|-------|---------------|-------------------|-------------|-------------|")
        w(f"| {jnr_c} | {jnr_n-jnr_c-jnr_np} | {jnr_np} | {jnr_sw} | {jnr_n} | "
          f"**{100*jnr_c/jnr_n:.1f}%** |")
        w()

    if all_team:
        combined_team = pd.concat(all_team)
        t_eval  = combined_team[combined_team.eval_status != "skipped"]
        t_clean = t_eval[t_eval.eval_status != "id_switch"]
        t_c  = (t_clean.eval_status == "correct").sum()
        t_n  = len(t_clean)
        t_sw = (t_eval.eval_status == "id_switch").sum()
        w(f"### Team / Team-side")
        w(f"| Correct | Wrong | ID-switch (excl.) | Total clean | **Accuracy** |")
        w(f"|---------|-------|-------------------|-------------|-------------|")
        w(f"| {t_c} | {t_n-t_c} | {t_sw} | {t_n} | **{100*t_c/t_n:.1f}%** |")
        w()

    if all_switches:
        combined_sw = pd.concat(all_switches)
        w(f"### ID Switches Summary")
        w(f"Total ID switches detected: **{len(combined_sw)}** "
          f"(across {combined_sw.video.nunique()} videos)")
        w()
        w("| Video | pred_tid | frames | purity | primary GT | predicted team |")
        w("|-------|----------|--------|--------|------------|----------------|")
        for _, r in combined_sw.iterrows():
            w(f"| {r.video} | {r.pred_tid} | {r.n_frames} | {r.purity} | "
              f"{r.primary} | {r.pred_team} |")
        w()
        w("> **Note**: ID switches are tracklets where the tracker jumps between two different GT players")
        w("> (purity < 0.80). The mean embedding/color of these tracklets is a mix of both players,")
        w("> making JNR and team classification unreliable. Root cause: tracker Re-ID failure.")

    report = "\n".join(lines)
    Path(out_path).write_text(report)
    print(f"\nReport saved: {out_path}")
    print(report)
    return report


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pklz", help="Path to .pklz state file")
    parser.add_argument("--gt_base", default=None,
                        help="GT base dir (auto-detected from video IDs if omitted)")
    parser.add_argument("--out", default=None,
                        help="Output .md path (default: report_<pklz_stem>.md next to pklz)")
    args = parser.parse_args()

    pklz_path = Path(args.pklz)

    # auto-detect valid vs test from video IDs
    if args.gt_base:
        gt_base = args.gt_base
    else:
        with zipfile.ZipFile(pklz_path) as zf:
            codes = [n.replace(".pkl","") for n in zf.namelist()
                     if n.endswith(".pkl") and not n.endswith("_image.pkl")
                     and n != "summary.json"]
        first_num = int(codes[0]) if codes else 0
        gt_base = GT_BASE_TEST if first_num >= 116 else GT_BASE_VALID

    out_path = args.out or str(pklz_path.parent / f"report_{pklz_path.stem}.md")

    build_report(str(pklz_path), gt_base, out_path)
