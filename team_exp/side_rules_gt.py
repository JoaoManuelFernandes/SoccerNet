"""Testa critérios de atribuição do LADO sobre a anotação de referência (agrupamento perfeito), nos 160 vídeos:
  mean   : média de x dos jogadores de cada equipa (critério da referência)
  median : mediana de x por equipa
  vote   : voto por tracklet (média de x de cada tracklet >0 → direita), como o 1.º nível da nossa cascata
  extreme: por frame, o jogador de campo mais à ESQUERDA pertence à equipa que defende a esquerda (lei do fora de jogo);
           voto por frame (esquerdo) + simétrico (direito)
  extreme2: idem, mas usando o 2.º mais à esquerda/direita (último defesa), mais robusto a um atacante isolado
Saída: por critério, n.º de vídeos em que a decisão contradiz a anotação."""
import json, glob, os, collections, numpy as np
DATA = "/home/joao/soccernet/data/SoccerNetGS"
def decide_from_scores(sl, sr):
    # sl[t]: pontuação "esquerda" da equipa t; devolve equipa que fica à esquerda
    return max(sl, key=lambda t: sl[t] - sr[t])
res = collections.defaultdict(list); n = 0
for split in ("test", "valid", "train"):
    for gtf in sorted(glob.glob(f"{DATA}/{split}/SNGS-*/Labels-GameState.json")):
        v = os.path.basename(os.path.dirname(gtf)); g = json.load(open(gtf))
        per_frame = collections.defaultdict(list); per_track = collections.defaultdict(list); teams = {}
        for a in g["annotations"]:
            at = a.get("attributes") or {}
            if at.get("role") != "player" or not a.get("bbox_pitch") or at.get("team") not in ("left", "right"): continue
            x = a["bbox_pitch"]["x_bottom_middle"]
            if abs(x) > 60: continue
            per_frame[a["image_id"]].append((x, at["team"])); per_track[a["track_id"]].append(x); teams[a["track_id"]] = at["team"]
        if len(set(teams.values())) < 2: continue
        n += 1; truth = "left"  # a equipa anotada 'left' está à esquerda por definição
        xs = {"left": [], "right": []}
        for tid, xl in per_track.items(): xs[teams[tid]] += xl
        # mean / median
        for name, f in (("mean", np.mean), ("median", np.median)):
            left_team = min(xs, key=lambda t: f(xs[t])); res[name].append((v, left_team == truth))
        # vote por tracklet
        votes = {"left": [0, 0], "right": [0, 0]}
        for tid, xl in per_track.items(): votes[teams[tid]][0 if np.mean(xl) < 0 else 1] += 1
        frac_left = {t: votes[t][0] / max(1, sum(votes[t])) for t in votes}
        left_team = max(frac_left, key=frac_left.get); res["vote"].append((v, left_team == truth))
        # extremos por frame
        for name, k in (("extreme", 0), ("extreme2", 1)):
            sl = {"left": 0, "right": 0}; sr = {"left": 0, "right": 0}
            for iid, lst in per_frame.items():
                if len(lst) < 4: continue
                srt = sorted(lst);
                if len(srt) > k: sl[srt[k][1]] += 1; sr[srt[-1 - k][1]] += 1
            left_team = decide_from_scores(sl, sr); res[name].append((v, left_team == truth))
print(f"vídeos avaliados: {n}")
for name, lst in res.items():
    bad = [v for v, ok in lst if not ok]
    print(f"{name:9s}: erros = {len(bad):2d}  {' '.join(b[5:] for b in bad)}")
