"""Simula o módulo de equipa completo (agrupamento + cascata de lado) nos 159 vídeos, a partir de
team_exp/hsv/<v>.npz e do ramo que o módulo real tomou (lido de nosso_out/logs/<v>.log), e devolve a
exatidão por tracklet da EQUIPA (o mesmo número que a tese reporta), por conjunto.

Variantes de agrupamento:
  atual : o team_cluster guardado no estado
  A     : descritor novo só onde o módulo real já usava cor (mesma arquitetura, mesmo gatilho)
  B     : descritor novo sempre (cor primeiro)
  C<t>  : descritor novo quando a sua silhueta >= t e não é degenerado; senão mantém o atual
A cascata de lado é a real (voto por tracklet → voto relativo → média ponderada por deteções).
"""
import glob, os, re, json, sys, numpy as np, pandas as pd, warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")
HSV = "/home/joao/soccernet/team_exp/hsv"; LOGS = "/home/joao/soccernet/nosso_out/logs"
DATA = "/home/joao/soccernet/data/SoccerNetGS"
VOTE, RELDIFF, MIN_FRAC = 0.60, 0.10, 0.20
FEAT = os.environ.get("FEAT", "hsv_torso")

def split_of(v):
    for s in ("test", "valid", "train"):
        if os.path.isdir(f"{DATA}/{s}/{v}"): return s

def branch_of(v):
    p = f"{LOGS}/{v}.log"
    if not os.path.exists(p): return None
    t = re.sub(r"\s+", " ", open(p, errors="ignore").read())
    m = re.search(r"team_cluster\][^\[]{0,400}?(color_tail|raw ReID)", t)
    return m.group(1) if m else None

def side_cascade(cl, mx, n):
    """cl,mx,n por tracklet -> dict cluster->'left'/'right' (cascata real)."""
    ok = ~np.isnan(mx)
    fr = {c: (mx[ok & (cl == c)] > 0).mean() if (ok & (cl == c)).any() else np.nan for c in (0, 1)}
    side = {c: ("right" if fr[c] >= VOTE else "left" if fr[c] <= 1 - VOTE else None) for c in (0, 1)}
    right = None
    if side[0] and side[1] and side[0] != side[1]: right = 0 if side[0] == "right" else 1
    elif side[0] and not side[1]: right = 0 if side[0] == "right" else 1
    elif side[1] and not side[0]: right = 1 if side[1] == "right" else 0
    if right is None and not any(np.isnan(list(fr.values()))):
        d = fr[0] - fr[1]
        if abs(d) >= RELDIFF: right = 0 if d > 0 else 1
    if right is None:
        w = {c: np.average(mx[ok & (cl == c)], weights=n[ok & (cl == c)]) if (ok & (cl == c)).any() else 0.0 for c in (0, 1)}
        right = 0 if w[0] > w[1] else 1
    return {right: "right", 1 - right: "left"}

def acc(team_pred, gt, ev):
    return (team_pred[ev] == gt[ev]).sum(), int(ev.sum())

def main():
    thr = [float(x) for x in (sys.argv[1:] or ["0.30", "0.40", "0.50"])]
    variants = ["atual", "A", "B"] + [f"C{t:.2f}" for t in thr]
    tot = {k: {} for k in variants}; per_video = {}
    for f in sorted(glob.glob(f"{HSV}/SNGS-*.npz")):
        v = os.path.basename(f)[:-4]; sp = split_of(v); d = np.load(f, allow_pickle=True)
        player = d["role"] == "player"; ev = d["evaluable"] & player; gt = d["gt_team"]
        if ev.sum() < 1 or player.sum() < 4: continue
        cur = np.nan_to_num(d["team_cluster"], nan=-1).astype(int)
        X = d[FEAT][player]
        km = KMeans(2, n_init=10, random_state=0).fit(X)
        new = np.full(len(gt), -1); new[player] = km.labels_
        sil = silhouette_score(X, km.labels_) if len(set(km.labels_)) == 2 else -1.0
        cnt = np.bincount(km.labels_, minlength=2); frac = cnt.min() / cnt.sum()
        br = branch_of(v); mx, n = d["mean_px"], np.maximum(d["n_det"], 1)
        row = {"split": sp, "sil_color": float(sil), "frac": float(frac), "branch": br}
        for k in variants:
            if k == "atual": cl = cur
            elif k == "A": cl = new if br == "color_tail" else cur
            elif k == "B": cl = new
            else:
                t = float(k[1:]); cl = new if (sil >= t and frac >= MIN_FRAC) else cur
            m = side_cascade(cl, mx, n)
            pred = np.array([m.get(c, "nan") for c in cl])
            ok, tot_n = acc(pred, gt, ev)
            row[k] = ok / tot_n if tot_n else np.nan
            t_ = tot[k].setdefault(sp, [0, 0]); t_[0] += ok; t_[1] += tot_n
        per_video[v] = row
    json.dump(per_video, open("/home/joao/soccernet/team_exp/simula_modulo.json", "w"), indent=1)
    print(f"característica: {FEAT}   vídeos: {len(per_video)}")
    print(f"{'variante':8s} " + " ".join(f"{s:>16s}" for s in ("test", "valid", "train")) + f"{'TOTAL':>16s}")
    for k in variants:
        cells = []; a = b = 0
        for s in ("test", "valid", "train"):
            x, y = tot[k].get(s, [0, 0]); a += x; b += y; cells.append(f"{100*x/y:14.1f}%" if y else "  —")
        print(f"{k:8s} " + " ".join(cells) + f"{100*a/b:15.1f}%")
    print("\nvídeos que mudam (atual vs B), ordenados pela diferença:")
    rows = sorted(per_video.items(), key=lambda kv: kv[1]["B"] - kv[1]["atual"])
    for v, r in rows:
        if abs(r["B"] - r["atual"]) > 0.02:
            print(f"  {v} {r['split']:5s} atual={r['atual']:.2f} A={r['A']:.2f} B={r['B']:.2f} sil={r['sil_color']:.2f} frac={r['frac']:.2f} ramo={r['branch']}")

if __name__ == "__main__":
    main()
