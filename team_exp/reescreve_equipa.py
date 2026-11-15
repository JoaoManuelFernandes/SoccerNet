"""Reescreve as colunas team_cluster e team nos estados de nosso_out/states, aplicando a auditoria de
agrupamento degenerado do módulo (grupo menor com < 8% das deteções de jogadores → reagrupar pela cor do
tronco; resgate só aceite se equilibrado) e, a seguir, a cascata de lado ORIGINAL do módulo.

Não altera mais nada no estado. Guarda cópia dos originais em nosso_out/states_prev/ na primeira passagem.
Uso: reescreve_equipa.py [--dry] [vídeos...]
"""
import sys, os, glob, json, zipfile, pickle, shutil, argparse, importlib.util, logging
import numpy as np, pandas as pd
logging.disable(logging.CRITICAL)
ROOT = "/home/joao/soccernet"; ST = f"{ROOT}/nosso_out/states"; PREV = f"{ROOT}/nosso_out/states_prev"
MOD = f"{ROOT}/sn-gamestate/sn_gamestate/team"

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
clu = load_mod("clu", f"{MOD}/tracklet_team_clustering_api.py")
sid = load_mod("sid", f"{MOD}/tracklet_team_side_labeling_api.py")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="*"); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(); os.makedirs(PREV, exist_ok=True)
    vids = a.videos or sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{ST}/SNGS-*.pklz"))
    rel = {}
    import pandas as _pd
    for v in vids:
        p = f"{ST}/{v}.pklz"; code = v[5:]
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            if f"{code}.pkl" not in names: print(f"{v}: sem deteções, ignorado", flush=True); continue
            blobs = {n: z.read(n) for n in names}
        det = pickle.loads(blobs[f"{code}.pkl"]); img = pickle.loads(blobs[f"{code}_image.pkl"])
        player = det[(det.role == "player") & det.track_id.notna()]
        if len(player) < 50 or player.team_cluster.isna().all():
            rel[v] = {"acao": "ignorado (poucos jogadores)"}; print(f"{v}: ignorado", flush=True); continue
        counts = np.array([(player.team_cluster == c).sum() for c in (0, 1)], dtype=float)
        frac = counts.min() / counts.sum() if counts.sum() else 0.0
        if frac >= clu._DEGENERATE_DET_FRAC:
            rel[v] = {"acao": "intacto", "frac": round(float(frac), 4)}; print(f"{v}: intacto (frac={frac:.3f})", flush=True); continue
        tids = sorted(player.track_id.unique())
        # guarda 1: árbitros a mais significa que falta uma equipa ao agrupamento
        n_ref = int(det[det.role == "referee"].dropna(subset=["track_id"]).track_id.nunique())
        ref_ratio = n_ref / len(tids) if tids else 0.0
        if ref_ratio > clu._MAX_REFEREE_RATIO:
            rel[v] = {"acao": "sem resgate (árbitros a mais)", "frac": round(float(frac), 4),
                      "arbitros": n_ref, "jogadores": len(tids)}
            print(f"{v}: SEM RESGATE — {n_ref} árbitros para {len(tids)} jogadores "
                  f"({ref_ratio:.0%}), falta uma equipa", flush=True); continue
        X = clu._tracklet_color_features(player, img, tids)
        if X is None:
            rel[v] = {"acao": "sem frames"}; print(f"{v}: sem frames, intacto", flush=True); continue
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
        cnt = np.bincount(km.labels_, minlength=2); f2 = cnt.min() / cnt.sum()
        # guarda 2: o resgate tem de passar o mesmo teste que reprovou a partição, nas deteções
        nd = np.array([(player.track_id == t).sum() for t in tids], dtype=float)
        dc = np.array([nd[km.labels_ == c].sum() for c in (0, 1)])
        d2 = dc.min() / dc.sum() if dc.sum() else 0.0
        if f2 < clu._RESCUE_MIN_FRAC or d2 < clu._RESCUE_MIN_FRAC:
            rel[v] = {"acao": "resgate rejeitado", "frac": round(float(frac), 4),
                      "frac_resgate": round(float(f2), 4), "frac_det_resgate": round(float(d2), 4)}
            print(f"{v}: resgate REJEITADO (tracklets {f2:.2f}, deteções {d2:.2f}) — intacto", flush=True); continue
        novo = dict(zip(tids, km.labels_))
        det2 = det.copy()
        mask = (det2.role == "player") & det2.track_id.notna()
        det2.loc[mask, "team_cluster"] = det2.loc[mask, "track_id"].map(novo).astype(float)
        det2.loc[mask, "team"] = np.nan
        det2 = sid.TrackletTeamSideLabeling().process(det2, img)
        mudou = int((det2.loc[mask, "team"] != det.loc[mask, "team"]).sum())
        rel[v] = {"acao": "reagrupado", "frac": round(float(frac), 4), "clusters": f"{cnt[0]}/{cnt[1]}",
                  "detecoes_mudadas": mudou, "de": f"{int(counts[0])}/{int(counts[1])}"}
        print(f"{v}: REAGRUPADO {int(counts[0])}/{int(counts[1])} -> {cnt[0]}/{cnt[1]} tracklets, "
              f"{mudou} deteções mudam de equipa", flush=True)
        if a.dry: continue
        if not os.path.exists(f"{PREV}/{v}.pklz"): shutil.copy2(p, f"{PREV}/{v}.pklz")
        blobs[f"{code}.pkl"] = pickle.dumps(det2)
        tmp = p + ".tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
            for n, b in blobs.items(): zo.writestr(n, b)
        os.replace(tmp, p)
    json.dump(rel, open(f"{ROOT}/team_exp/reescrita.json", "w"), indent=1, ensure_ascii=False)
    n = sum(1 for r in rel.values() if r.get("acao") == "reagrupado")
    print(f"\n{len(rel)} vídeos processados; {n} reagrupados" + ("  (simulação, nada gravado)" if a.dry else ""))

if __name__ == "__main__":
    main()
