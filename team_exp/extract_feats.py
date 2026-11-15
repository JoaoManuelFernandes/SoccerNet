"""Extrai, por vídeo, características alternativas por tracklet para experiências de atribuição de equipa,
sem tocar na pipeline. Lê o estado do nosso sistema (nosso_out/states), recorta os jogadores nas imagens
originais e calcula, para uma amostra de deteções por tracklet:
  hsv_torso  (24) histograma HSV do tronco, sem píxeis de relva
  lab_torso  (3)  Lab médio do tronco, sem relva
  clip_full / clip_torso (512)  CLIP ViT-B/32 (openai)
  dino_full / dino_torso (384)  DINOv2-small (CLS)
Guarda team_exp/feats/<video>.npz com as médias por tracklet, o rótulo GT (via generate_report.analyse_video),
a equipa/cluster atuais e a função prevista.
Uso: extract_feats.py SNGS-025 [SNGS-027 ...]   (K deteções por tracklet: --k 24)
"""
import sys, os, zipfile, pickle, argparse, json
import numpy as np, pandas as pd, cv2, torch
sys.path.insert(0, "/home/joao/soccernet/sn-gamestate")
from generate_report import analyse_video

ROOT = "/home/joao/soccernet"; DATA = f"{ROOT}/data/SoccerNetGS"; OUT = f"{ROOT}/team_exp/feats"
os.makedirs(OUT, exist_ok=True)

def split_of(v):
    for s in ("test", "valid", "train"):
        if os.path.isdir(f"{DATA}/{s}/{v}"): return s
    raise FileNotFoundError(v)

def load_state(v):
    z = zipfile.ZipFile(f"{ROOT}/nosso_out/states/{v}.pklz")
    code = [n for n in z.namelist() if n.endswith(".pkl") and not n.endswith("_image.pkl")][0][:-4]
    det = pickle.loads(z.read(f"{code}.pkl")); img = pickle.loads(z.read(f"{code}_image.pkl"))
    return code, det, img

def torso_box(x, y, w, h):
    return int(x + 0.2 * w), int(y + 0.12 * h), max(2, int(0.6 * w)), max(2, int(0.40 * h))

def grass_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= 30) & (h <= 90) & (s >= 50) & (vv >= 40)
    return (~green).astype(np.uint8), hsv

def color_feats(bgr):
    if bgr.size == 0 or bgr.shape[0] < 2 or bgr.shape[1] < 2:
        return np.zeros(24, np.float32), np.zeros(3, np.float32)
    mask, hsv = grass_mask(bgr)
    if mask.sum() < 8: mask = np.ones(bgr.shape[:2], np.uint8)
    parts = []
    for ch, hi in ((0, 180), (1, 256), (2, 256)):
        hst = cv2.calcHist([hsv], [ch], mask, [8], [0, hi]).flatten().astype(np.float32)
        parts.append(hst / hst.sum() if hst.sum() > 0 else hst)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[mask > 0].mean(0).astype(np.float32) / 255.0
    return np.concatenate(parts), lab

class Models:
    def __init__(self, dev):
        from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor
        self.dev = dev
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
        self.clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
        self.dino_p = AutoImageProcessor.from_pretrained("facebook/dinov2-small", use_fast=True)
    @torch.no_grad()
    def embed(self, rgb_list):
        out = {}
        px = self.clip_p(images=rgb_list, return_tensors="pt")["pixel_values"].to(self.dev)
        e = self.clip.get_image_features(pixel_values=px); out["clip"] = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        px = self.dino_p(images=rgb_list, return_tensors="pt")["pixel_values"].to(self.dev)
        e = self.dino(pixel_values=px).last_hidden_state[:, 0]; out["dino"] = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; models = Models(dev)
    for v in a.videos:
        if os.path.exists(f"{OUT}/{v}.npz"): print(v, "já existe"); continue
        split = split_of(v); code, det, img = load_state(v)
        path_of = dict(zip(img["id"].astype(str) if "id" in img.columns else img.index.astype(str), img["file_path"]))
        # GT por tracklet (mesmo critério da avaliação por tracklet da tese)
        jnr, team, _ = analyse_video(code, det, f"{DATA}/{split}")
        gt = {int(r.pred_tid): r for r in team.itertuples()} if team is not None else {}
        d = det[det.track_id.notna()].copy(); d["track_id"] = d.track_id.astype(int)
        # amostra de K deteções por tracklet, uniformemente no tempo
        rows = []
        for tid, g in d.groupby("track_id"):
            g = g.sort_values("image_id")
            idx = np.linspace(0, len(g) - 1, min(a.k, len(g))).round().astype(int)
            rows.append(g.iloc[np.unique(idx)])
        samp = pd.concat(rows); samp["image_id"] = samp.image_id.astype(str)
        feats = {k: [] for k in ("hsv_torso", "lab_torso", "clip_full", "clip_torso", "dino_full", "dino_torso")}
        order = []  # (tid) por deteção amostrada
        buf_full, buf_torso, buf_meta = [], [], []
        def flush():
            if not buf_full: return
            ef = models.embed(buf_full); et = models.embed(buf_torso)
            feats["clip_full"].extend(ef["clip"]); feats["dino_full"].extend(ef["dino"])
            feats["clip_torso"].extend(et["clip"]); feats["dino_torso"].extend(et["dino"])
            buf_full.clear(); buf_torso.clear()
        for iid, g in samp.groupby("image_id", sort=True):
            fp = path_of.get(iid)
            if fp is None: continue
            im = cv2.imread(fp if os.path.isabs(fp) else f"{DATA}/{split}/{v}/{fp}")
            if im is None: continue
            H, W = im.shape[:2]
            for r in g.itertuples():
                x, y, w, h = [float(t) for t in r.bbox_ltwh]
                x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h))
                if x1 - x0 < 4 or y1 - y0 < 8: continue
                full = im[y0:y1, x0:x1]
                tx, ty, tw, th = torso_box(x, y, w, h); tx0, ty0 = max(0, tx), max(0, ty); tx1, ty1 = min(W, tx + tw), min(H, ty + th)
                torso = im[ty0:ty1, tx0:tx1] if (tx1 - tx0 >= 2 and ty1 - ty0 >= 2) else full
                hs, lb = color_feats(torso); feats["hsv_torso"].append(hs); feats["lab_torso"].append(lb)
                buf_full.append(cv2.cvtColor(full, cv2.COLOR_BGR2RGB)); buf_torso.append(cv2.cvtColor(torso, cv2.COLOR_BGR2RGB))
                order.append(int(r.track_id))
                if len(buf_full) >= a.batch: flush()
        flush()
        order = np.array(order); tids = np.unique(order)
        out = {"tids": tids}
        for k, val in feats.items():
            M = np.vstack(val); out[k] = np.vstack([M[order == t].mean(0) for t in tids])
        # rótulos e estado atual por tracklet
        first = d.groupby("track_id").first()
        out["role"] = np.array([str(first.loc[t, "role"]) for t in tids])
        out["team_cluster"] = np.array([float(first.loc[t, "team_cluster"]) if pd.notna(first.loc[t, "team_cluster"]) else np.nan for t in tids])
        out["team"] = np.array([str(first.loc[t, "team"]) for t in tids])
        out["jersey"] = np.array([str(first.loc[t, "jersey_number"]) if "jersey_number" in first.columns else "nan" for t in tids])
        out["n_det"] = np.array([int((d.track_id == t).sum()) for t in tids])
        out["gt_team"] = np.array([str(gt[t].gt_team) if t in gt else "nan" for t in tids])
        out["gt_role"] = np.array([str(gt[t].gt_role) if t in gt else "nan" for t in tids])
        out["evaluable"] = np.array([bool(t in gt and gt[t].eval_status in ("correct", "wrong")) for t in tids])
        out["mean_px"] = np.array([np.nanmean([b["x_bottom_middle"] for b in d[d.track_id == t].bbox_pitch if isinstance(b, dict) and abs(b.get("x_bottom_middle", 999)) <= 52.5] or [np.nan]) for t in tids])
        np.savez_compressed(f"{OUT}/{v}.npz", **out)
        ev = out["evaluable"]; cur = (out["team"] == out["gt_team"])[ev].mean() if ev.any() else float("nan")
        print(f"{v} ({split}): {len(tids)} tracklets, {len(order)} recortes, avaliáveis={ev.sum()}, equipa atual={cur:.3f}", flush=True)

if __name__ == "__main__":
    main()
