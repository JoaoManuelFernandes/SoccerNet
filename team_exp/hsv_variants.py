"""Variantes do descritor de cor do tronco, avaliadas por agrupamento (k-means, melhor permutação) nos vídeos dados.
  grass   : máscara fixa de verde (H 30-90, S>=50, V>=40)                      [= hsv_torso atual]
  none    : sem máscara
  local   : máscara pela cor de fundo LOCAL (mediana HSV das colunas à esquerda/direita da caixa, nas linhas do tronco)
  local_fb: idem, com recuo para 'sem máscara' quando sobra < 30% dos píxeis
  grass_fb: máscara fixa com o mesmo recuo
Uso: hsv_variants.py SNGS-069 SNGS-025 ..."""
import sys, os, numpy as np, pandas as pd, cv2, warnings
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/joao/soccernet/team_exp"); from extract_feats import load_state, torso_box
sys.path.insert(0, "/home/joao/soccernet/team_exp"); from cluster_eval import best_perm_acc
HSV_DIR = "/home/joao/soccernet/team_exp/hsv"

def hist(hsv, mask):
    parts = []
    for ch, hi in ((0, 180), (1, 256), (2, 256)):
        h = cv2.calcHist([hsv], [ch], mask, [8], [0, hi]).flatten().astype(np.float32); parts.append(h / h.sum() if h.sum() > 0 else h)
    return np.concatenate(parts)

def masks(im, x, y, w, h):
    H, W = im.shape[:2]; tx, ty, tw, th = torso_box(x, y, w, h)
    tx0, ty0, tx1, ty1 = max(0, tx), max(0, ty), min(W, tx + tw), min(H, ty + th)
    if tx1 - tx0 < 2 or ty1 - ty0 < 2: return None
    torso = im[ty0:ty1, tx0:tx1]; hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    grass = ((hh >= 30) & (hh <= 90) & (ss >= 50) & (vv >= 40))
    # fundo local: colunas fora da caixa (margem de 0.3w de cada lado), mesmas linhas do tronco
    m = max(2, int(0.3 * w)); lx0, lx1 = max(0, int(x) - m), max(0, int(x)); rx0, rx1 = min(W, int(x + w)), min(W, int(x + w) + m)
    bg = [im[ty0:ty1, lx0:lx1], im[ty0:ty1, rx0:rx1]]; bg = [b for b in bg if b.size]
    if bg:
        bgh = cv2.cvtColor(np.hstack([cv2.resize(b, (max(1, b.shape[1]), ty1 - ty0)) for b in bg]), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        med = np.median(bgh, 0); dh = np.abs(hh - med[0]); dh = np.minimum(dh, 180 - dh)
        local = (dh <= 8) & (np.abs(ss - med[1]) <= 50) & (np.abs(vv - med[2]) <= 50)
    else: local = grass
    out = {"none": np.ones(hh.shape, np.uint8), "grass": (~grass).astype(np.uint8), "local": (~local).astype(np.uint8)}
    for k in ("grass", "local"):
        mk = out[k]; out[k + "_fb"] = mk if mk.mean() >= 0.30 else out["none"]
        if mk.sum() < 8: out[k] = out["none"]
    return hsv, out

def main():
    vids = sys.argv[1:]; K = 24; res = []
    for v in vids:
        p = f"{HSV_DIR}/{v}.npz"; p = p if os.path.exists(p) else f"/home/joao/soccernet/team_exp/feats/{v}.npz"
        d = np.load(p, allow_pickle=True); code, det, img = load_state(v); path_of = dict(zip(img["id"].astype(str), img["file_path"]))
        det = det[det.track_id.notna()].copy(); det["track_id"] = det.track_id.astype(int)
        rows = []
        for tid, g in det.groupby("track_id"):
            g = g.sort_values("image_id"); idx = np.linspace(0, len(g) - 1, min(K, len(g))).round().astype(int); rows.append(g.iloc[np.unique(idx)])
        samp = pd.concat(rows); samp["image_id"] = samp.image_id.astype(str)
        feats = {k: [] for k in ("none", "grass", "local", "grass_fb", "local_fb")}; order = []
        for iid, g in samp.groupby("image_id", sort=True):
            im = cv2.imread(path_of.get(iid, ""))
            if im is None: continue
            for r in g.itertuples():
                x, y, w, h = [float(t) for t in r.bbox_ltwh]; mm = masks(im, x, y, w, h)
                if mm is None: continue
                hsv, ms = mm
                for k in feats: feats[k].append(hist(hsv, ms[k]))
                order.append(int(r.track_id))
        order = np.array(order); tids = d["tids"]; player = d["role"] == "player"; ev = d["evaluable"] & player; gt = d["gt_team"]
        row = {"v": v, "atual": best_perm_acc(np.nan_to_num(d["team_cluster"], nan=-1).astype(int), gt, ev)}
        for k, val in feats.items():
            M = np.vstack(val); X = np.vstack([M[order == t].mean(0) if (order == t).any() else np.zeros(24) for t in tids])
            lab = np.full(len(tids), -1); lab[player] = KMeans(2, n_init=10, random_state=0).fit_predict(X[player]); row[k] = best_perm_acc(lab, gt, ev)
        res.append(row); print(row, flush=True)
    df = pd.DataFrame(res); print("\nmédias:"); print(df.drop(columns="v").mean().round(3).to_string())

if __name__ == "__main__":
    main()
