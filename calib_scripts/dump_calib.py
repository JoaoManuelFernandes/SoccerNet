"""
Extrai dados de calibração team-side de um pklz para NPZ compactos por vídeo.
Uso: python dump_calib.py <pklz> <out_dir>
Por vídeo: por deteção (track_id, image_id, bbox_ltwh, pitch_x/y, role) +
embeddings_with_colors médio por tracklet.
Carrega UM vídeo de cada vez (RAM 7.6GB).
"""
import pickle, sys, zipfile
from pathlib import Path
import numpy as np

pklz, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(pklz) as zf:
    codes = [n.replace(".pkl", "") for n in zf.namelist()
             if n.endswith(".pkl") and not n.endswith("_image.pkl")]
    for code in sorted(codes):
        out = out_dir / f"SNGS-{code}.npz"
        if out.exists():
            print(f"skip {code} (existe)")
            continue
        with zf.open(f"{code}.pkl") as f:
            df = pickle.load(f)

        # colunas por deteção
        cols = {}
        cols["track_id"] = df["track_id"].to_numpy(dtype=np.float64)
        cols["image_id"] = df["image_id"].to_numpy(dtype=np.int64)
        cols["bbox_ltwh"] = np.stack(df["bbox_ltwh"].to_numpy())
        pitch = df["bbox_pitch"].apply(
            lambda p: (p.get("x_bottom_middle", np.nan), p.get("y_bottom_middle", np.nan))
            if isinstance(p, dict) else (np.nan, np.nan))
        cols["pitch_xy"] = np.array(pitch.tolist(), dtype=np.float64)
        role_col = "role" if "role" in df.columns else "role_detection"
        cols["role"] = df[role_col].astype(str).to_numpy()

        # embeddings_with_colors médio por tracklet (o que o clustering usa)
        tids, embs = [], []
        if "embeddings_with_colors" in df.columns:
            emb_df = df.dropna(subset=["track_id"])
            for tid, g in emb_df.groupby("track_id"):
                vecs = [np.asarray(v, dtype=np.float32) for v in g["embeddings_with_colors"]
                        if v is not None and np.size(v) > 1]
                if vecs:
                    tids.append(float(tid))
                    embs.append(np.mean(vecs, axis=0))
        cols["emb_tids"] = np.array(tids, dtype=np.float64)
        cols["emb_means"] = np.stack(embs) if embs else np.zeros((0, 283), dtype=np.float32)

        np.savez_compressed(out, **cols)
        print(f"OK {code}: {len(df)} dets, {len(tids)} tracklets c/ emb, "
              f"{out.stat().st_size/1e6:.1f}MB")
        del df, cols
