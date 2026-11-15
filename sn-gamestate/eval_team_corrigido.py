"""
Atribuição de equipa: métrica ANTIGA vs CORRIGIDA.

Reutiliza as funções do eval_team_by_tracklet.py original. A ÚNICA diferença
está em quem entra na avaliação:

  ANTIGA:    det[det.role == "player"]   -> filtra pela função que NÓS prevemos.
             Um jogador que classificámos como árbitro desaparece da conta.
  CORRIGIDA: todas as deteções. O filtro é só pela função da ANOTAÇÃO.
             Esse jogador passa a contar como erro.
"""
import pickle, zipfile, glob, os
import numpy as np, pandas as pd
from eval_team_by_tracklet import (build_gt_tracks, match_predicted_to_gt,
                                   MIN_FRAMES, MATCH_RATIO_MIN, PURITY_MIN)

def avalia(code, pklz, gt_base, usar_role_previsto):
    gt_tracks, frame_anns = build_gt_tracks(f"{gt_base}/SNGS-{code}/Labels-GameState.json")
    with zipfile.ZipFile(pklz) as zf:
        det = pickle.load(zf.open(f"{code}.pkl"))

    dets = det[det.role == "player"].copy() if usar_role_previsto else det.copy()
    matches = match_predicted_to_gt(dets, frame_anns)

    rows = []
    for ptid, c in matches.items():
        gtid, nm = c.most_common(1)[0]
        gi = gt_tracks.get(gtid, {})
        sub = dets[dets.track_id == ptid]
        n = len(sub)
        if n == 0: continue
        teams = sub["team"].dropna() if "team" in sub.columns else pd.Series(dtype=object)
        pred = teams.mode()[0] if len(teams) else None     # None = ficou sem equipa
        rows.append({"gt_role": gi.get("role"), "gt_team": gi.get("team"),
                     "pred_team": pred, "n_frames": n,
                     "match_ratio": nm/n,
                     "purity": nm/sum(c.values()) if sum(c.values()) else 0.0})
    df = pd.DataFrame(rows)
    if df.empty: return 0, 0, 0
    v = df[(df.gt_role == "player") & (df.match_ratio >= MATCH_RATIO_MIN) &
           (df.n_frames >= MIN_FRAMES) & (df.purity >= PURITY_MIN) & df.gt_team.notna()]
    sem_equipa = int(v.pred_team.isna().sum())
    return int((v.pred_team == v.gt_team).sum()), len(v), sem_equipa

if __name__ == "__main__":
    gt_base = "/home/joao/soccernet/data/SoccerNetGS/test"
    estados = {}
    for p in glob.glob("/home/joao/soccernet/**/states/*.pklz", recursive=True):
        if os.path.getsize(p) < 1_000_000: continue
        try:
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if not n.endswith(".pkl"): continue
                    v = n[:-4]
                    if not (v.isdigit() and len(v) == 3): continue
                    if not os.path.isdir(f"{gt_base}/SNGS-{v}"): continue
                    if v not in estados or os.path.getmtime(p) > os.path.getmtime(estados[v]):
                        estados[v] = p
        except Exception: pass

    print(f"{'vídeo':<7}{'ANTIGA':>15}{'CORRIGIDA':>15}{'+tracklets':>11}{'s/equipa':>10}")
    print("-" * 60)
    ca=ta=cb=tb=se=0
    for v in sorted(estados):
        try:
            c1,t1,_  = avalia(v, estados[v], gt_base, True)
            c2,t2,s2 = avalia(v, estados[v], gt_base, False)
        except Exception as e:
            print(f"{v:<7} erro: {type(e).__name__}: {e}"); continue
        if t1 == 0 and t2 == 0: continue
        p1 = 100*c1/t1 if t1 else float('nan')
        p2 = 100*c2/t2 if t2 else float('nan')
        mark = "  <<<" if (t2-t1) > 0 else ""
        print(f"{v:<7}{c1:>4}/{t1:<4}{p1:>5.1f}%{c2:>5}/{t2:<4}{p2:>5.1f}%{t2-t1:>11}{s2:>10}{mark}")
        ca+=c1; ta+=t1; cb+=c2; tb+=t2; se+=s2
    print("-" * 60)
    if ta and tb:
        print(f"{'GLOBAL':<7}{ca:>4}/{ta:<4}{100*ca/ta:>5.1f}%{cb:>5}/{tb:<4}{100*cb/tb:>5.1f}%{tb-ta:>11}{se:>10}")
