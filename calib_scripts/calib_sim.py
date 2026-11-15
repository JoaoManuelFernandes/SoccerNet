"""
Simulador fiel do team clustering + side labeling sobre os dumps de calibração.
Reproduz exactamente a lógica de tracklet_team_clustering_api.py e
tracklet_team_side_labeling_api.py, parametrizável para grid-search.

Uso:
  validar:      python calib_sim.py validate
  grid-search:  python calib_sim.py grid
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DUMPS = {
    "test":  Path("/home/joao/soccernet/calib_dumps/test"),
    "valid": Path("/home/joao/soccernet/calib_dumps/valid"),
}
GT_BASE = {
    "test":  Path("/home/joao/soccernet/data/SoccerNetGS/test"),
    "valid": Path("/home/joao/soccernet/data/SoccerNetGS/valid"),
}
CACHE = Path("/home/joao/soccernet/calib_dumps/prep_cache")
CACHE.mkdir(exist_ok=True)

# ── parâmetros por defeito (código actual) ──────────────────────────────────
DEFAULT = dict(
    ONE_SIDED_THRESH=0.85,
    REID_WEAK_L2=2.04,
    VOTE_THRESHOLD=0.60,
    RELATIVE_VOTE_MIN_DIFF=0.25,
    MEAN_X_GAP_MIN=0.5,
    DRIFT_MIN_FRAMES=10,
)
_PITCH_X_MAX = 52.5
_MAX_VALID_PITCH_X = 60.0
_COLOR_TAIL_DIM = 27


# ── IoU matching contra GT (mesma lógica do generate_report.py) ─────────────
def iou_ltwh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    inter = max(0, min(ax+aw, bx+bw) - max(ax, bx)) * max(0, min(ay+ah, by+bh) - max(ay, by))
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0


def load_gt_tracks(gt_path):
    gt = json.load(open(gt_path))
    frame_anns = defaultdict(list)
    gt_team = {}
    for ann in gt["annotations"]:
        if "attributes" not in ann or "bbox_image" not in ann:
            continue
        a, b = ann["attributes"], ann["bbox_image"]
        frame_anns[str(ann["image_id"])].append(
            {"tid": ann["track_id"], "bbox": [b["x"], b["y"], b["w"], b["h"]]})
        gt_team[ann["track_id"]] = a.get("team")
    return frame_anns, gt_team


def match_tracklets(dump, frame_anns, gt_team):
    """pred_tid → (gt_team, n_frames, purity). Filtros do generate_report:
    min_frames=5, match_ratio≥0.3, purity≥0.8."""
    tids = dump["track_id"]; iids = dump["image_id"]; boxes = dump["bbox_ltwh"]
    per_tid_gt = defaultdict(list)
    for i in range(len(tids)):
        tid = tids[i]
        if np.isnan(tid):
            continue
        anns = frame_anns.get(str(iids[i]))
        if not anns:
            continue
        best, best_iou = None, 0.30
        for ann in anns:
            v = iou_ltwh(boxes[i], ann["bbox"])
            if v > best_iou:
                best, best_iou = ann["tid"], v
        per_tid_gt[tid].append(best)  # None se sem match

    result = {}
    for tid, matches in per_tid_gt.items():
        n = len(matches)
        if n < 5:
            continue
        matched = [m for m in matches if m is not None]
        if len(matched) / n < 0.30:
            continue
        vals, counts = np.unique(matched, return_counts=True)
        top_gt, top_n = vals[np.argmax(counts)], counts.max()
        purity = top_n / len(matched)
        if purity < 0.80:
            continue
        team = gt_team.get(top_gt)
        if team in ("left", "right"):
            result[tid] = team
    return result


# ── preparação por vídeo (cache) ────────────────────────────────────────────
def prep_video(split, vid):
    cache_f = CACHE / f"{split}_{vid}.npz"
    if cache_f.exists():
        d = np.load(cache_f, allow_pickle=True)
        return d["data"].item()
    dump = np.load(DUMPS[split] / f"{vid}.npz", allow_pickle=True)
    frame_anns, gt_team = load_gt_tracks(GT_BASE[split] / vid / "Labels-GameState.json")
    gt_map = match_tracklets(dump, frame_anns, gt_team)

    tids = dump["track_id"]; iids = dump["image_id"]
    px = dump["pitch_xy"][:, 0]; roles = dump["role"]
    emb_tids = dump["emb_tids"]; emb_means = dump["emb_means"]
    emb_by_tid = {t: emb_means[i] for i, t in enumerate(emb_tids)}

    # role por tracklet (constante pós-agg) + séries pitch_x ordenadas
    tr = {}
    order = np.argsort(iids, kind="stable")
    for i in order:
        tid = tids[i]
        if np.isnan(tid):
            continue
        e = tr.setdefault(tid, {"role": roles[i], "px": []})
        if not np.isnan(px[i]):
            e["px"].append(float(px[i]))

    players, gk_x = [], []
    for tid, e in tr.items():
        if e["role"] == "goalkeeper":
            gk_x.extend(e["px"])
        if e["role"] != "player" or tid not in emb_by_tid:
            continue
        arr = np.array(e["px"])
        arr_c = arr[np.abs(arr) <= _PITCH_X_MAX]          # filtro clustering
        arr_s = arr[np.abs(arr) <= _MAX_VALID_PITCH_X]    # filtro side-labeling
        players.append({
            "tid": tid,
            "emb_reid": emb_by_tid[tid][:256],
            "emb_color_tail": emb_by_tid[tid][-_COLOR_TAIL_DIM:],
            "mean_px_cluster": float(arr_c.mean()) if len(arr_c) >= 3 else np.nan,
            "mean_px_side": float(arr_s.mean()) if len(arr_s) else np.nan,
            "px_series": arr_s.tolist(),
            "gt": gt_map.get(tid),
        })
    gk_side = None
    if gk_x:
        gk_side = "right" if float(np.mean(gk_x)) > 0 else "left"

    data = {"players": players, "gk_side": gk_side,
            "gk_tracklets": {tid: ("right" if np.mean(e["px"]) > 0 else "left")
                             for tid, e in tr.items()
                             if e["role"] == "goalkeeper" and e["px"] and gt_map.get(tid)},
            "gk_gt": {tid: gt_map[tid] for tid, e in tr.items()
                      if e["role"] == "goalkeeper" and gt_map.get(tid)}}
    np.savez_compressed(cache_f, data=data)
    return data


# ── simulação com parâmetros ────────────────────────────────────────────────
def simulate_video(data, P):
    players = data["players"]
    if not players:
        return None
    X_reid = np.vstack([p["emb_reid"] for p in players])
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_reid)
    c0 = X_reid[km.labels_ == 0].mean(0); c1 = X_reid[km.labels_ == 1].mean(0)
    reid_l2 = float(np.linalg.norm(c0 - c1))

    px_valid = np.array([p["mean_px_cluster"] for p in players])
    px_valid = px_valid[~np.isnan(px_valid)]
    one_sided = len(px_valid) >= 6 and (
        (px_valid < 0).mean() >= P["ONE_SIDED_THRESH"] or
        (px_valid > 0).mean() >= P["ONE_SIDED_THRESH"])

    if one_sided and reid_l2 < P["REID_WEAK_L2"]:
        X = np.vstack([p["emb_color_tail"] for p in players])
        labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit(
            StandardScaler().fit_transform(X)).labels_
        branch = "color_tail"
    else:
        labels = km.labels_
        branch = "raw_reid"

    # ── side labeling chain ──
    stats = [(int(labels[i]), p["mean_px_side"], p["px_series"], p["gt"], p["tid"])
             for i, p in enumerate(players)]

    def tracklet_vote():
        votes = {0: {"r": 0, "l": 0}, 1: {"r": 0, "l": 0}}
        for cid, mx, _, _, _ in stats:
            if np.isnan(mx):
                continue
            votes[cid]["r" if mx > 0 else "l"] += 1
        out = {}
        for cid, v in votes.items():
            tot = v["r"] + v["l"]
            if tot == 0:
                continue
            r = v["r"] / tot
            if r >= P["VOTE_THRESHOLD"]:
                out[cid] = "right"
            elif r <= 1 - P["VOTE_THRESHOLD"]:
                out[cid] = "left"
        if not out:
            return None
        if len(out) == 1:
            cid, side = next(iter(out.items()))
            out[1 - cid] = "right" if side == "left" else "left"
        if len(set(out.values())) < 2:
            return None
        return out

    def relative_vote():
        rf = {}
        for cid in (0, 1):
            t = [mx for c, mx, _, _, _ in stats if c == cid and not np.isnan(mx)]
            if len(t) < 3:
                continue
            rf[cid] = float(np.mean(np.array(t) > 0))
        if len(rf) < 2:
            return None
        diff = rf[0] - rf[1]
        if abs(diff) < P["RELATIVE_VOTE_MIN_DIFF"]:
            return None
        return {0: "right", 1: "left"} if diff > 0 else {0: "left", 1: "right"}

    def gk_drift(gk_side):
        gk_sign = 1.0 if gk_side == "right" else -1.0
        drifts = {0: [], 1: []}
        for cid, _, series, _, _ in stats:
            if len(series) < P["DRIFT_MIN_FRAMES"]:
                continue
            drifts[cid].append(series[-1] - series[0])
        if len(drifts[0]) < 3 or len(drifts[1]) < 3:
            return None
        toward = {c: float(np.mean(drifts[c])) * gk_sign for c in (0, 1)}
        atk = max(toward, key=toward.get)
        away = "left" if gk_side == "right" else "right"
        return {atk: away, 1 - atk: gk_side}

    def mean_x_fallback():
        xa = [x for c, _, s, _, _ in stats if c == 0 for x in s]
        xb = [x for c, _, s, _, _ in stats if c == 1 for x in s]
        avg_a = float(np.mean(xa)) if xa else 0.0
        avg_b = float(np.mean(xb)) if xb else 0.0
        if abs(avg_a - avg_b) < P["MEAN_X_GAP_MIN"] and data["gk_side"]:
            r = gk_drift(data["gk_side"])
            if r:
                return r, "gk_drift"
        return ({0: "right", 1: "left"} if avg_a > avg_b
                else {0: "left", 1: "right"}), "mean_x"

    mapping = tracklet_vote(); method = "tracklet_vote"
    if mapping is None:
        mapping = relative_vote(); method = "relative_vote"
    if mapping is None:
        mapping, method = mean_x_fallback()

    # ── avaliação: tracklets player com GT + GKs (atribuição directa) ──
    correct = total = 0
    for cid, _, _, gt, _ in stats:
        if gt is None:
            continue
        total += 1
        if mapping[cid] == gt:
            correct += 1
    for tid, side in data["gk_tracklets"].items():
        gt = data["gk_gt"].get(tid)
        if gt is None:
            continue
        total += 1
        if side == gt:
            correct += 1
    return {"correct": correct, "total": total, "branch": branch,
            "method": method, "acc": correct / total if total else np.nan}


def run_all(P, splits=("test", "valid")):
    out = {}
    for split in splits:
        for f in sorted(DUMPS[split].glob("SNGS-*.npz")):
            vid = f.stem
            data = prep_video(split, vid)
            r = simulate_video(data, P)
            if r:
                out[(split, vid)] = r
    return out


# ── grid-search com cache de clustering ─────────────────────────────────────
_CLUSTER_CACHE = {}

def cluster_cache(split, vid, data):
    key = (split, vid)
    if key in _CLUSTER_CACHE:
        return _CLUSTER_CACHE[key]
    players = data["players"]
    X_reid = np.vstack([p["emb_reid"] for p in players])
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_reid)
    c0 = X_reid[km.labels_ == 0].mean(0); c1 = X_reid[km.labels_ == 1].mean(0)
    X_col = np.vstack([p["emb_color_tail"] for p in players])
    labels_color = KMeans(n_clusters=2, random_state=0, n_init=10).fit(
        StandardScaler().fit_transform(X_col)).labels_
    px = np.array([p["mean_px_cluster"] for p in players])
    px = px[~np.isnan(px)]
    entry = {
        "labels_raw": km.labels_, "labels_color": labels_color,
        "reid_l2": float(np.linalg.norm(c0 - c1)),
        "frac_left": float((px < 0).mean()) if len(px) else 0.0,
        "frac_right": float((px > 0).mean()) if len(px) else 0.0,
        "n_px": len(px),
    }
    _CLUSTER_CACHE[key] = entry
    return entry


def simulate_cached(data, cc, P):
    players = data["players"]
    one_sided = cc["n_px"] >= 6 and max(cc["frac_left"], cc["frac_right"]) >= P["ONE_SIDED_THRESH"]
    if one_sided and cc["reid_l2"] < P["REID_WEAK_L2"]:
        labels, branch = cc["labels_color"], "color_tail"
    else:
        labels, branch = cc["labels_raw"], "raw_reid"

    stats = [(int(labels[i]), p["mean_px_side"], p["px_series"], p["gt"])
             for i, p in enumerate(players)]

    votes = {0: {"r": 0, "l": 0}, 1: {"r": 0, "l": 0}}
    for cid, mx, _, _ in stats:
        if not np.isnan(mx):
            votes[cid]["r" if mx > 0 else "l"] += 1
    mapping, method = None, None
    out = {}
    for cid, v in votes.items():
        tot = v["r"] + v["l"]
        if tot == 0:
            continue
        r = v["r"] / tot
        if r >= P["VOTE_THRESHOLD"]:
            out[cid] = "right"
        elif r <= 1 - P["VOTE_THRESHOLD"]:
            out[cid] = "left"
    if out:
        if len(out) == 1:
            cid, side = next(iter(out.items()))
            out[1 - cid] = "right" if side == "left" else "left"
        if len(set(out.values())) == 2:
            mapping, method = out, "tracklet_vote"

    if mapping is None:
        rf = {}
        for cid in (0, 1):
            t = [mx for c, mx, _, _ in stats if c == cid and not np.isnan(mx)]
            if len(t) >= 3:
                rf[cid] = float(np.mean(np.array(t) > 0))
        if len(rf) == 2 and abs(rf[0] - rf[1]) >= P["RELATIVE_VOTE_MIN_DIFF"]:
            mapping = {0: "right", 1: "left"} if rf[0] > rf[1] else {0: "left", 1: "right"}
            method = "relative_vote"

    if mapping is None:
        xa = [x for c, _, s, _ in stats if c == 0 for x in s]
        xb = [x for c, _, s, _ in stats if c == 1 for x in s]
        avg_a = float(np.mean(xa)) if xa else 0.0
        avg_b = float(np.mean(xb)) if xb else 0.0
        method = "mean_x"
        if abs(avg_a - avg_b) < P["MEAN_X_GAP_MIN"] and data["gk_side"]:
            gk_sign = 1.0 if data["gk_side"] == "right" else -1.0
            drifts = {0: [], 1: []}
            for cid, _, series, _ in stats:
                if len(series) >= P["DRIFT_MIN_FRAMES"]:
                    drifts[cid].append(series[-1] - series[0])
            if len(drifts[0]) >= 3 and len(drifts[1]) >= 3:
                toward = {c: float(np.mean(drifts[c])) * gk_sign for c in (0, 1)}
                atk = max(toward, key=toward.get)
                away = "left" if data["gk_side"] == "right" else "right"
                mapping, method = {atk: away, 1 - atk: data["gk_side"]}, "gk_drift"
        if mapping is None:
            mapping = ({0: "right", 1: "left"} if avg_a > avg_b
                       else {0: "left", 1: "right"})

    correct = total = 0
    for cid, _, _, gt in stats:
        if gt is None:
            continue
        total += 1
        if mapping[cid] == gt:
            correct += 1
    for tid, side in data["gk_tracklets"].items():
        gt = data["gk_gt"].get(tid)
        if gt is not None:
            total += 1
            if side == gt:
                correct += 1
    return correct, total


def grid_search():
    import itertools
    # carregar tudo uma vez
    videos = {}
    for split in ("test", "valid"):
        for f in sorted(DUMPS[split].glob("SNGS-*.npz")):
            vid = f.stem
            data = prep_video(split, vid)
            if data["players"]:
                videos[(split, vid)] = (data, cluster_cache(split, vid, data))
    print(f"{len(videos)} vídeos carregados "
          f"({sum(1 for s,_ in videos if s=='test')} test, "
          f"{sum(1 for s,_ in videos if s=='valid')} valid)")

    # baseline com params actuais (por vídeo, para a regra de zero regressões)
    base = {k: simulate_cached(d, c, DEFAULT) for k, (d, c) in videos.items()}
    bt_c = sum(c for (s, _), (c, t) in base.items() if s == "test")
    bt_t = sum(t for (s, _), (c, t) in base.items() if s == "test")
    bv_c = sum(c for (s, _), (c, t) in base.items() if s == "valid")
    bv_t = sum(t for (s, _), (c, t) in base.items() if s == "valid")
    print(f"BASELINE sim: test {bt_c}/{bt_t}={100*bt_c/bt_t:.1f}%  "
          f"valid {bv_c}/{bv_t}={100*bv_c/bv_t:.1f}%")

    grid = dict(
        ONE_SIDED_THRESH=[0.75, 0.80, 0.85, 0.90],
        REID_WEAK_L2=[1.8, 2.04, 2.2, 2.4, 2.7, 3.0],
        VOTE_THRESHOLD=[0.55, 0.60, 0.65, 0.70],
        RELATIVE_VOTE_MIN_DIFF=[0.15, 0.25, 0.35],
        MEAN_X_GAP_MIN=[0.3, 0.5, 1.0, 2.0, 4.5],
        DRIFT_MIN_FRAMES=[10],
    )
    keys = list(grid)
    results = []
    for combo in itertools.product(*grid.values()):
        P = dict(zip(keys, combo))
        test_c = test_t = val_c = val_t = 0
        regression = False
        per_video = {}
        for k, (d, c) in videos.items():
            corr, tot = simulate_cached(d, c, P)
            per_video[k] = (corr, tot)
            if k[0] == "test":
                test_c += corr; test_t += tot
                if corr < base[k][0]:          # zero regressões por vídeo no test
                    regression = True
                    break
            else:
                val_c += corr; val_t += tot
        if regression:
            continue
        results.append((val_c, test_c, P, per_video))

    results.sort(key=lambda r: (-r[0], -r[1]))
    print(f"\n{len(results)} combos sem regressão no test. Top 10 por valid:")
    for val_c, test_c, P, _ in results[:10]:
        print(f"  valid {val_c}/{bv_t} ({100*val_c/bv_t:.1f}%)  "
              f"test {test_c}/{bt_t} ({100*test_c/bt_t:.1f}%)  {P}")

    if results:
        val_c, test_c, P, pv = results[0]
        print("\n=== VENCEDOR — mudanças por vídeo (valid) vs baseline ===")
        for k in sorted(videos):
            if k[0] != "valid":
                continue
            b, n = base[k], pv[k]
            if n[0] != b[0]:
                d = "MELHOR" if n[0] > b[0] else "PIOR"
                print(f"  {k[1]}: {b[0]}/{b[1]} → {n[0]}/{n[1]}  [{d}]")
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":
        res = run_all(DEFAULT, splits=("test",))
        tc = sum(r["correct"] for r in res.values())
        tt = sum(r["total"] for r in res.values())
        print(f"TEST global (params actuais): {tc}/{tt} = {100*tc/tt:.1f}%")
        for (s, v), r in sorted(res.items()):
            flag = "" if r["acc"] >= 0.999 else "  ← <100%"
            print(f"  {v}: {r['acc']*100:5.1f}% ({r['correct']}/{r['total']}) "
                  f"[{r['branch']}/{r['method']}]{flag}")
    elif mode == "grid":
        grid_search()
