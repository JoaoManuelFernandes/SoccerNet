"""Testa critérios de atribuição do LADO sobre os CLUSTERS PREVISTOS (nosso_out/states), nos 159 vídeos.
Para cada vídeo: mapeamento correto = cada cluster vai para a equipa GT maioritária dos seus tracklets avaliáveis
(só conta quando os dois clusters têm maiorias GT distintas e pureza ≥ 0.7, i.e. quando o lado é decidível).
Critérios: cascata atual (voto → voto relativo → média), média, mediana, média ponderada por deteções,
voto por tracklet só, voto→média (cascata sem o voto relativo), voto→mediana.
Guarda team_exp/side_rules_pred.json e imprime erros por critério."""
import sys, zipfile, pickle, glob, os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/joao/soccernet/sn-gamestate"); from generate_report import analyse_video
DATA = "/home/joao/soccernet/data/SoccerNetGS"; N = "/home/joao/soccernet/nosso_out/states"
VOTE, RELDIFF, GAP = 0.60, 0.10, 0.3

def split_of(v):
    for s in ("test", "valid", "train"):
        if os.path.isdir(f"{DATA}/{s}/{v}"): return s

def rules(ts):
    """ts: DataFrame por tracklet (cl, mx, n). Devolve dict critério -> cluster que fica à DIREITA (0/1) ou None."""
    out = {}
    def by_stat(f):
        a = f(ts[ts.cl == 0]); b = f(ts[ts.cl == 1]); return 0 if a > b else 1
    out["mean"] = by_stat(lambda s: s.mx.mean()); out["median"] = by_stat(lambda s: s.mx.median())
    out["wmean"] = by_stat(lambda s: np.average(s.mx, weights=s.n))
    # voto por tracklet
    fr = {c: (ts[ts.cl == c].mx > 0).mean() for c in (0, 1)}
    side = {c: ("right" if fr[c] >= VOTE else "left" if fr[c] <= 1 - VOTE else None) for c in (0, 1)}
    vote = None
    if side[0] and side[1] and side[0] != side[1]: vote = 0 if side[0] == "right" else 1
    elif side[0] and not side[1]: vote = 0 if side[0] == "right" else 1
    elif side[1] and not side[0]: vote = 1 if side[1] == "right" else 0
    out["vote_only"] = vote
    # cascata atual: voto → voto relativo → média
    rel = None
    if vote is None:
        d = fr[0] - fr[1]
        if abs(d) >= RELDIFF: rel = 0 if d > 0 else 1
    out["atual"] = vote if vote is not None else (rel if rel is not None else out["wmean"])   # módulo real: média sobre DETEÇÕES
    out["fr0"] = float(fr[0]); out["fr1"] = float(fr[1]); out["rel"] = rel
    # variantes: voto → wmean com desempate pela fração só quando a diferença de médias é pequena
    a = np.average(ts[ts.cl == 0].mx, weights=ts[ts.cl == 0].n); b = np.average(ts[ts.cl == 1].mx, weights=ts[ts.cl == 1].n)
    out["wmean_gap"] = float(abs(a - b))
    out["vote_wmean_then_rel"] = vote if vote is not None else ((0 if a > b else 1) if abs(a - b) >= GAP else (rel if rel is not None else (0 if a > b else 1)))
    out["vote_mean"] = vote if vote is not None else out["mean"]
    out["vote_median"] = vote if vote is not None else out["median"]
    out["vote_wmean"] = vote if vote is not None else out["wmean"]
    # média com desempate pela fração quando a diferença de médias é pequena
    a = ts[ts.cl == 0].mx.mean(); b = ts[ts.cl == 1].mx.mean()
    out["mean_then_frac"] = (0 if a > b else 1) if abs(a - b) >= GAP else (0 if fr[0] > fr[1] else 1)
    return out

res = {}
for split in ("test", "valid", "train"):
    for f in sorted(glob.glob(f"{N}/SNGS-*.pklz")):
        v = os.path.basename(f)[:-5]
        if split_of(v) != split: continue
        code = v[5:]; z = zipfile.ZipFile(f)
        if f"{code}.pkl" not in z.namelist(): continue
        det = pickle.loads(z.read(f"{code}.pkl"))
        _, team, _ = analyse_video(code, det, f"{DATA}/{split}")
        if team is None: continue
        gt = {int(r.pred_tid): (r.gt_team, r.eval_status) for r in team.itertuples()}
        p = det[(det.role == "player") & det.track_id.notna() & det.team_cluster.notna()].copy()
        p["px"] = p.bbox_pitch.apply(lambda b: b["x_bottom_middle"] if isinstance(b, dict) and abs(b.get("x_bottom_middle", 999)) <= 60 else np.nan)
        ts = p.groupby("track_id").agg(cl=("team_cluster", "first"), tm=("team", "first"), mx=("px", "mean"), n=("px", "count")).reset_index().dropna(subset=["mx"])
        ts["cl"] = ts.cl.astype(int); ts["gtt"] = ts.track_id.astype(int).map(lambda t: gt.get(t, ("?", "?"))[0])
        ts["ev"] = ts.track_id.astype(int).map(lambda t: gt.get(t, ("?", "?"))[1] in ("correct", "wrong"))
        if (ts.cl == 0).sum() < 2 or (ts.cl == 1).sum() < 2: continue
        maj = {}; pur = {}
        for c in (0, 1):
            e = ts[(ts.cl == c) & ts.ev]
            if len(e) == 0: maj[c] = None; continue
            vc = e.gtt.value_counts(); maj[c] = vc.index[0]; pur[c] = vc.iloc[0] / len(e)
        decidable = maj.get(0) and maj.get(1) and maj[0] != maj[1] and min(pur.values()) >= 0.7
        r = rules(ts); right_true = 0 if maj.get(0) == "right" else 1
        res[v] = {"split": split, "decidable": bool(decidable), "right_true": right_true, "rules": r,
                  "one_sided": bool(((ts.mx < 0).mean() >= 0.85) or ((ts.mx > 0).mean() >= 0.85)),
                  "atual_team_ok": float(np.mean((ts.tm == ts.gtt)[ts.ev])) if ts.ev.any() else None}
        print(v, "decidível" if decidable else "indecidível", {k: ("ok" if val == right_true else "ERRO" if val is not None else "-") for k, val in r.items() if k not in ("fr0","fr1","rel","wmean_gap")}, flush=True)
json.dump(res, open("/home/joao/soccernet/team_exp/side_rules_pred.json", "w"), indent=1)
dec = {v: r for v, r in res.items() if r["decidable"]}
print(f"\nvídeos: {len(res)}, decidíveis: {len(dec)}")
for k in [k for k in next(iter(dec.values()))["rules"] if k not in ("fr0","fr1","rel","wmean_gap")]:
    bad = [v[5:] for v, r in dec.items() if r["rules"][k] is not None and r["rules"][k] != r["right_true"]]
    und = [v[5:] for v, r in dec.items() if r["rules"][k] is None]
    print(f"{k:15s}: erros={len(bad):2d} {' '.join(bad)}" + (f"  | sem decisão={len(und)}" if und else ""))
