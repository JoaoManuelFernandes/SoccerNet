"""
Avaliador OFFLINE do approach A (voto ponderado por qualidade) para JNR.
Lê o pklz (bbox/track/image_id + GT via IoU-match) e o jnr_results.json
(historial + window_meta) de uma corrida com save_debug_crops=true, e compara:
  - baseline: voto por contagem (Counter, como o módulo faz hoje)
  - A: voto ponderado por qualidade da janela (n_crops × sharpness × área)
sem re-correr o Qwen.

Uso: python jnr_vote_ab.py <exp_dir> <eval_set>
  <exp_dir> = /home/joao/soccernet/jnr_exp/<TAG>  (tem jnrexp-*.pklz e jnr_v2_debug/)
"""
import json, sys, zipfile, pickle
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

TRACK_MIN_AGREEMENT = 0.45   # igual ao módulo
MIN_VALID_WINDOWS   = 1

GT_BASE = {
    "test":  Path("/home/joao/soccernet/data/SoccerNetGS/test"),
    "valid": Path("/home/joao/soccernet/data/SoccerNetGS/valid"),
}


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    inter = max(0, min(ax+aw, bx+bw)-max(ax, bx)) * max(0, min(ay+ah, by+bh)-max(ay, by))
    u = aw*ah + bw*bh - inter
    return inter/u if u > 0 else 0.0


def load_gt(gt_path):
    gt = json.load(open(gt_path))
    frame_anns = defaultdict(list); gt_jersey = {}
    for ann in gt["annotations"]:
        if "attributes" not in ann or "bbox_image" not in ann:
            continue
        a, b = ann["attributes"], ann["bbox_image"]
        frame_anns[str(ann["image_id"])].append(
            {"tid": ann["track_id"], "bbox": [b["x"], b["y"], b["w"], b["h"]]})
        j = a.get("jersey")
        gt_jersey[ann["track_id"]] = str(j) if j is not None else None
    return frame_anns, gt_jersey


def match_tracklets(df, frame_anns, gt_jersey):
    """pred_tid → gt_jersey (str) para tracklets com jersey no GT, purity≥0.8, ≥5 frames."""
    per = defaultdict(list)
    for _, r in df.iterrows():
        tid = r["track_id"]
        if tid is None or (isinstance(tid, float) and np.isnan(tid)):
            continue
        anns = frame_anns.get(str(r["image_id"]))
        if not anns:
            continue
        box = r["bbox_ltwh"]
        best, biou = None, 0.30
        for ann in anns:
            v = iou(box, ann["bbox"])
            if v > biou:
                best, biou = ann["tid"], v
        per[tid].append(best)
    out = {}
    for tid, ms in per.items():
        n = len(ms)
        if n < 5:
            continue
        matched = [m for m in ms if m is not None]
        if not matched or len(matched)/n < 0.30:
            continue
        vals, cnts = np.unique(matched, return_counts=True)
        top = vals[np.argmax(cnts)]
        if cnts.max()/len(matched) < 0.80:
            continue
        j = gt_jersey.get(top)
        if j is not None and j != "None":
            out[tid] = j
    return out


def vote_count(historial):
    valid = [r for r in historial if r not in (None, "None")]
    if len(valid) < MIN_VALID_WINDOWS:
        return None
    c = Counter(valid); best, n = c.most_common(1)[0]
    return best if n/len(historial) >= TRACK_MIN_AGREEMENT else None


def vote_weighted(historial, meta, alpha_sharp=1.0, alpha_area=1.0):
    """Approach A: peso por janela = n_crops * (mean_sharp^a) * (mean_area^b) normalizado.
    None contribui com o seu peso CONTRA (para o denominador), como no count-vote."""
    if not meta or len(meta) != len(historial):
        return vote_count(historial)  # fallback se meta desalinhada
    weights = []
    for m in meta:
        w = max(m.get("n_crops", 0), 1) \
            * (max(m.get("mean_sharp", 1.0), 1.0) ** alpha_sharp) \
            * (max(m.get("mean_area", 1.0), 1.0) ** alpha_area)
        weights.append(w)
    total = sum(weights)
    if total <= 0:
        return vote_count(historial)
    score = defaultdict(float)
    for r, w in zip(historial, weights):
        if r not in (None, "None"):
            score[r] += w
    if not score:
        return None
    best = max(score, key=score.get)
    return best if score[best]/total >= TRACK_MIN_AGREEMENT else None


def evaluate(exp_dir, evalset, schemes):
    exp = Path(exp_dir)
    pklz = next(exp.glob("jnrexp-*.pklz"))
    with zipfile.ZipFile(pklz) as zf:
        codes = [n[:-4] for n in zf.namelist() if n.endswith(".pkl") and not n.endswith("_image.pkl")]
        results = {}
        for code in sorted(codes):
            vid = f"SNGS-{code}"
            with zf.open(f"{code}.pkl") as f:
                df = pickle.load(f)
            frame_anns, gt_jersey = load_gt(GT_BASE[evalset] / vid / "Labels-GameState.json")
            gt_map = match_tracklets(df, frame_anns, gt_jersey)

            jf = exp / "jnr_v2_debug" / vid / "jnr_results.json"
            if not jf.exists():
                # alguns dumps ficam em jnr_v2_debug/ root sem subpasta
                cand = list((exp / "jnr_v2_debug").glob(f"**/{vid}*/jnr_results.json"))
                jf = cand[0] if cand else None
            hist = {}
            if jf and jf.exists():
                for e in json.load(open(jf)):
                    hist[float(e["tid"])] = (e.get("historial", []), e.get("window_meta", []))

            per_scheme = {name: [0, 0] for name in schemes}  # correct, total
            for tid, gtj in gt_map.items():
                h, m = hist.get(tid, ([], []))
                for name, fn in schemes.items():
                    pred = fn(h, m)
                    per_scheme[name][1] += 1
                    if pred is not None and str(pred) == str(gtj):
                        per_scheme[name][0] += 1
            results[vid] = per_scheme
    return results


if __name__ == "__main__":
    exp_dir, evalset = sys.argv[1], sys.argv[2]
    schemes = {
        "baseline(count)": lambda h, m: vote_count(h),
        "A:sharp1_area1":  lambda h, m: vote_weighted(h, m, 1.0, 1.0),
        "A:sharp0_area1":  lambda h, m: vote_weighted(h, m, 0.0, 1.0),
        "A:sharp1_area0":  lambda h, m: vote_weighted(h, m, 1.0, 0.0),
        "A:sharp0.5_area0.5": lambda h, m: vote_weighted(h, m, 0.5, 0.5),
    }
    res = evaluate(exp_dir, evalset, schemes)
    names = list(schemes)
    print(f"{'Vídeo':<12} " + " ".join(f"{n:<18}" for n in names))
    tot = {n: [0, 0] for n in names}
    for vid in sorted(res):
        row = f"{vid:<12} "
        for n in names:
            c, t = res[vid][n]; tot[n][0] += c; tot[n][1] += t
            row += f"{c}/{t} ({100*c/t if t else 0:.0f}%)".ljust(19)
        print(row)
    print("-"*80)
    row = f"{'GLOBAL':<12} "
    for n in names:
        c, t = tot[n]; row += f"{c}/{t} ({100*c/t if t else 0:.1f}%)".ljust(19)
    print(row)
