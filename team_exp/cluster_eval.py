"""Avalia, sobre team_exp/feats/*.npz, a qualidade do AGRUPAMENTO em duas equipas (independente do lado):
exatidão por tracklet avaliável com a melhor permutação cluster→equipa. Compara o atual (team_cluster do
estado) com alternativas de características e de algoritmo. Uso: cluster_eval.py [--group failures|guard|all]"""
import glob, os, sys, argparse, warnings
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
F = "/home/joao/soccernet/team_exp/feats"
FAIL_GRUPO = {"SNGS-025","SNGS-027","SNGS-028","SNGS-030","SNGS-036","SNGS-038","SNGS-069","SNGS-091","SNGS-131"}
FAIL_LADO = {"SNGS-106","SNGS-113"}
FUNC = {"SNGS-039","SNGS-042","SNGS-045","SNGS-048","SNGS-052","SNGS-057"}

def best_perm_acc(labels, gt, ev):
    labels, gt = labels[ev], gt[ev]
    if len(gt) == 0: return np.nan
    a = np.mean((labels == 0) == (gt == "left")); b = np.mean((labels == 1) == (gt == "left"))
    return max(a, b)

def feats_of(d, name):
    if "+" in name:
        parts = [feats_of(d, p) for p in name.split("+")]
        return np.hstack([StandardScaler().fit_transform(p) / np.sqrt(p.shape[1]) for p in parts])
    X = d[name].astype(np.float64)
    return X

METHODS = {
    "km":      lambda X: KMeans(2, n_init=10, random_state=0).fit_predict(X),
    "km_std":  lambda X: KMeans(2, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(X)),
    "agg_cos": lambda X: AgglomerativeClustering(2, metric="cosine", linkage="average").fit_predict(X),
    "agg_ward":lambda X: AgglomerativeClustering(2, linkage="ward").fit_predict(X),
    "gmm":     lambda X: GaussianMixture(2, covariance_type="diag", random_state=0, n_init=3).fit(X).predict(X),
    "spec":    lambda X: SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=min(10, max(2, len(X)-1)), random_state=0).fit_predict(X),
    "km_pca8": lambda X: KMeans(2, n_init=10, random_state=0).fit_predict(PCA(min(8, X.shape[1], len(X)-1)).fit_transform(StandardScaler().fit_transform(X))),
}
FEATS = ["lab_torso", "hsv_torso", "clip_full", "clip_torso", "dino_full", "dino_torso", "clip_torso+hsv_torso", "dino_torso+hsv_torso", "clip_torso+lab_torso", "clip_full+clip_torso"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--group", default="all"); ap.add_argument("--methods", default="km,km_std,agg_cos,agg_ward,gmm,spec,km_pca8")
    ap.add_argument("--min_det", type=int, default=1); a = ap.parse_args()
    methods = a.methods.split(",")
    files = sorted(glob.glob(f"{F}/SNGS-*.npz"))
    res = {}; cur = {}; curc = {}
    for f in files:
        v = os.path.basename(f)[:-4]; d = np.load(f, allow_pickle=True)
        grp = "grupo" if v in FAIL_GRUPO else "lado" if v in FAIL_LADO else "func" if v in FUNC else "guard"
        if a.group != "all" and grp != a.group: continue
        player = d["role"] == "player"; ev = d["evaluable"] & player
        if ev.sum() < 3: continue
        gt = d["gt_team"]
        cur[v] = np.mean((d["team"] == gt)[ev]); tc = np.nan_to_num(d["team_cluster"], nan=-1).astype(int)
        curc[v] = best_perm_acc(tc, gt, ev)
        for fn in FEATS:
            X = feats_of(d, fn)[player]
            for m in methods:
                try: lab = METHODS[m](X)
                except Exception as e: lab = np.zeros(len(X), int)
                full = np.full(len(d["tids"]), -1); full[player] = lab
                res.setdefault((fn, m), {})[v] = best_perm_acc(full, gt, ev)
    vids = sorted(cur)
    def grp_of(v): return "grupo" if v in FAIL_GRUPO else "lado" if v in FAIL_LADO else "func" if v in FUNC else "guard"
    groups = sorted(set(grp_of(v) for v in vids))
    print(f"vídeos: {len(vids)}  ({', '.join(f'{g}={sum(grp_of(v)==g for v in vids)}' for g in groups)})")
    print(f"{'características':28s} {'método':9s} " + " ".join(f"{g:>7s}" for g in groups) + "   média  ≥95%  <70%")
    def row(name, m, sc):
        vals = [sc.get(v, np.nan) for v in vids]
        per = " ".join(f"{np.nanmean([sc[v] for v in vids if grp_of(v)==g]):7.3f}" for g in groups)
        print(f"{name:28s} {m:9s} {per}   {np.nanmean(vals):.3f}  {sum(x>=0.95 for x in vals):4d}  {sum(x<0.70 for x in vals):4d}")
    row("ATUAL (team vs gt)", "-", cur); row("ATUAL só agrupamento", "-", curc)
    rows = sorted(res.items(), key=lambda kv: -np.nanmean([kv[1].get(v, np.nan) for v in vids]))
    for (fn, m), sc in rows: row(fn, m, sc)
    # detalhe por vídeo dos 3 melhores + atual
    print("\nPor vídeo (só agrupamento, melhor permutação):")
    top = [k for k, _ in rows[:4]]
    print(f"{'vídeo':9s} {'grp':6s} {'atual':>6s} " + " ".join(f"{fn[:14]+'/'+m:>22s}" for fn, m in top))
    for v in vids:
        print(f"{v:9s} {grp_of(v):6s} {curc[v]:6.2f} " + " ".join(f"{res[k].get(v, np.nan):22.2f}" for k in top))

if __name__ == "__main__":
    main()
