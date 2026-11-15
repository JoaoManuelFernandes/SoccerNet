"""Versão rápida (só CPU) do extractor: histograma HSV do tronco sem relva (24-d) + Lab médio (3-d), K recortes por
tracklet, para TODOS os tracklets (todas as funções) de cada vídeo em nosso_out/states. Guarda team_exp/hsv/<v>.npz
com: tids, hsv_torso, lab_torso, role, team_cluster, team, gt_team, gt_role, evaluable, n_det, mean_px.
Uso: extract_hsv_all.py [--k 24] [vídeos...]  (sem vídeos = todos os que faltam)"""
import sys, os, glob, argparse, numpy as np, pandas as pd, cv2
sys.path.insert(0, "/home/joao/soccernet/team_exp"); sys.path.insert(0, "/home/joao/soccernet/sn-gamestate")
from extract_feats import split_of, load_state, torso_box, color_feats, DATA, ROOT
from generate_report import analyse_video
OUT = f"{ROOT}/team_exp/hsv"; os.makedirs(OUT, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="*"); ap.add_argument("--k", type=int, default=24); a = ap.parse_args()
    vids = a.videos or sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{ROOT}/nosso_out/states/SNGS-*.pklz"))
    for v in vids:
        if os.path.exists(f"{OUT}/{v}.npz"): continue
        try: split = split_of(v); code, det, img = load_state(v)
        except Exception as e: print(v, "sem estado:", e, flush=True); continue
        if "track_id" not in det.columns: print(v, "sem tracks", flush=True); continue
        path_of = dict(zip(img["id"].astype(str), img["file_path"]))
        _, team, _ = analyse_video(code, det, f"{DATA}/{split}")
        gt = {int(r.pred_tid): r for r in team.itertuples()} if team is not None else {}
        d = det[det.track_id.notna()].copy(); d["track_id"] = d.track_id.astype(int)
        rows = []
        for tid, g in d.groupby("track_id"):
            g = g.sort_values("image_id"); idx = np.linspace(0, len(g) - 1, min(a.k, len(g))).round().astype(int)
            rows.append(g.iloc[np.unique(idx)])
        samp = pd.concat(rows); samp["image_id"] = samp.image_id.astype(str)
        hs_l, lb_l, order = [], [], []
        for iid, g in samp.groupby("image_id", sort=True):
            fp = path_of.get(iid); im = cv2.imread(fp) if fp else None
            if im is None: continue
            H, W = im.shape[:2]
            for r in g.itertuples():
                x, y, w, h = [float(t) for t in r.bbox_ltwh]
                tx, ty, tw, th = torso_box(x, y, w, h); tx0, ty0 = max(0, tx), max(0, ty); tx1, ty1 = min(W, tx + tw), min(H, ty + th)
                if tx1 - tx0 < 2 or ty1 - ty0 < 2: continue
                hs, lb = color_feats(im[ty0:ty1, tx0:tx1]); hs_l.append(hs); lb_l.append(lb); order.append(int(r.track_id))
        order = np.array(order); tids = np.unique(order); HS = np.vstack(hs_l); LB = np.vstack(lb_l)
        first = d.groupby("track_id").first()
        out = {"tids": tids, "hsv_torso": np.vstack([HS[order == t].mean(0) for t in tids]), "lab_torso": np.vstack([LB[order == t].mean(0) for t in tids]),
               "role": np.array([str(first.loc[t, "role"]) for t in tids]),
               "team_cluster": np.array([float(first.loc[t, "team_cluster"]) if pd.notna(first.loc[t, "team_cluster"]) else np.nan for t in tids]),
               "team": np.array([str(first.loc[t, "team"]) for t in tids]),
               "n_det": np.array([int((d.track_id == t).sum()) for t in tids]),
               "gt_team": np.array([str(gt[t].gt_team) if t in gt else "nan" for t in tids]),
               "gt_role": np.array([str(gt[t].gt_role) if t in gt else "nan" for t in tids]),
               "evaluable": np.array([bool(t in gt and gt[t].eval_status in ("correct", "wrong")) for t in tids])}
        px = d.assign(px=d.bbox_pitch.apply(lambda b: b["x_bottom_middle"] if isinstance(b, dict) and abs(b.get("x_bottom_middle", 999)) <= 60 else np.nan)).groupby("track_id").px.mean()
        out["mean_px"] = np.array([float(px.get(t, np.nan)) for t in tids])
        np.savez_compressed(f"{OUT}/{v}.npz", **out)
        ev = out["evaluable"]; print(f"{v} ({split}): {len(tids)} tracklets, {len(order)} recortes, avaliáveis={ev.sum()}", flush=True)

if __name__ == "__main__":
    main()
