"""Grelha agrupamento x cascata de lado, nos 159 vídeos, com a exatidão por tracklet da equipa (número da tese).
Agrupamento: atual (do estado) | novo (k-means no histograma HSV do tronco)
Cascata: variantes que diferem no nível do voto relativo e no filtro de tracklets fiáveis para as estatísticas de posição.
Motivo: o voto relativo dispara em assimetrias de 10 pontos percentuais na fração de tracklets à direita e sobrepõe-se
à média ponderada, que é muito mais fiável; tracklets curtos (poucas deteções) entram nessa fração com o mesmo peso
que um tracklet de 700 deteções."""
import glob, os, json, sys, numpy as np, warnings
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
HSV = "/home/joao/soccernet/team_exp/hsv"; DATA = "/home/joao/soccernet/data/SoccerNetGS"
VOTE, RELDIFF = 0.60, 0.10

def split_of(v):
    for s in ("test", "valid", "train"):
        if os.path.isdir(f"{DATA}/{s}/{v}"): return s

def cascade(cl, mx, n, use_rel=True, min_det=0, reldiff=RELDIFF, weight_vote=False):
    ok = (~np.isnan(mx)) & (n >= min_det)
    if not ok.any(): ok = ~np.isnan(mx)
    fr = {}
    for c in (0, 1):
        m = ok & (cl == c)
        if not m.any(): fr[c] = np.nan; continue
        fr[c] = float(np.average(mx[m] > 0, weights=n[m])) if weight_vote else float((mx[m] > 0).mean())
    side = {c: ("right" if fr[c] >= VOTE else "left" if fr[c] <= 1 - VOTE else None) if not np.isnan(fr[c]) else None for c in (0, 1)}
    right = None
    if side[0] and side[1] and side[0] != side[1]: right = 0 if side[0] == "right" else 1
    elif side[0] and not side[1]: right = 0 if side[0] == "right" else 1
    elif side[1] and not side[0]: right = 1 if side[1] == "right" else 0
    if right is None and use_rel and not any(np.isnan(list(fr.values()))):
        d = fr[0] - fr[1]
        if abs(d) >= reldiff: right = 0 if d > 0 else 1
    if right is None:
        w = {}
        for c in (0, 1):
            m = ok & (cl == c); w[c] = float(np.average(mx[m], weights=n[m])) if m.any() else 0.0
        right = 0 if w[0] > w[1] else 1
    return {right: "right", 1 - right: "left"}

CASC = {
    "atual":            dict(use_rel=True,  min_det=0),
    "sem_rel":          dict(use_rel=False, min_det=0),
    "rel0.25":          dict(use_rel=True,  min_det=0, reldiff=0.25),
    "min30":            dict(use_rel=True,  min_det=30),
    "sem_rel+min30":    dict(use_rel=False, min_det=30),
    "sem_rel+min60":    dict(use_rel=False, min_det=60),
    "voto_pond":        dict(use_rel=True,  min_det=0, weight_vote=True),
    "voto_pond+semrel": dict(use_rel=False, min_det=0, weight_vote=True),
    "vp+semrel+min30":  dict(use_rel=False, min_det=30, weight_vote=True),
}

def main():
    FEAT = os.environ.get("FEAT", "hsv_torso")
    res = {}; per_video = {}
    for f in sorted(glob.glob(f"{HSV}/SNGS-*.npz")):
        v = os.path.basename(f)[:-4]; sp = split_of(v); d = np.load(f, allow_pickle=True)
        pl = d["role"] == "player"; ev = d["evaluable"] & pl; gt = d["gt_team"]
        if ev.sum() < 1 or pl.sum() < 4: continue
        cur = np.nan_to_num(d["team_cluster"], nan=-1).astype(int)
        new = np.full(len(gt), -1); new[pl] = KMeans(2, n_init=10, random_state=0).fit_predict(d[FEAT][pl])
        mx, n = d["mean_px"], np.maximum(d["n_det"], 1)
        pv = {"split": sp}
        for gname, cl in (("atual", cur), ("novo", new)):
            for cname, cfg in CASC.items():
                m = cascade(cl, mx, n, **cfg); pred = np.array([m.get(c, "nan") for c in cl])
                ok = int((pred[ev] == gt[ev]).sum()); t = res.setdefault((gname, cname), {}).setdefault(sp, [0, 0])
                t[0] += ok; t[1] += int(ev.sum()); pv[f"{gname}/{cname}"] = ok / max(1, int(ev.sum()))
        per_video[v] = pv
    json.dump(per_video, open("/home/joao/soccernet/team_exp/simula_grid.json", "w"), indent=1)
    print(f"característica: {FEAT}   vídeos: {len(per_video)}\n")
    print(f"{'agrup.':7s} {'cascata':18s} " + " ".join(f"{s:>8s}" for s in ("test", "valid", "train")) + f"{'TOTAL':>9s}")
    rows = []
    for (g, c), t in res.items():
        cells = []; a = b = 0
        for s in ("test", "valid", "train"):
            x, y = t.get(s, [0, 0]); a += x; b += y; cells.append(f"{100*x/y:7.1f}%" if y else "    —")
        rows.append((100 * a / b, g, c, cells))
    for tot, g, c, cells in sorted(rows, reverse=True):
        print(f"{g:7s} {c:18s} " + " ".join(cells) + f"{tot:8.1f}%")

if __name__ == "__main__":
    main()
