"""Como compara_final.py, mas a coluna 'nosso' é o sistema final: estados da tese com o número decidido pela
confiança da leitura (gshota/conf_decisoes.json). 'orig' e 'base_jnrv2' vêm da cache de compara_final.py (inalterados).
Saída: nosso_out/COMPARACAO_FINAL_conf.md (mesmo formato)."""
import sys, json, os, re
import numpy as np
from pathlib import Path
sys.path.insert(0, "/home/joao/soccernet/nosso_out/tools")
import importlib.util
N = Path("/home/joao/soccernet/nosso_out")
DEC = json.load(open(N/"gshota"/"conf_decisoes.json"))
spec = importlib.util.spec_from_file_location("cf", N/"tools"/"compara_final.py")
src = open(N/"tools"/"compara_final.py").read()
# reaproveita as funções sem correr o corpo do script
cab = src.split("rows = []")[0]
g = {}; exec(compile(cab, "compara_final_head", "exec"), g)
cache = g["cache"]; SPLITS = g["SPLITS"]
from generate_report import analyse_video
CC = N/"tools"/"compara_final_conf_cache.json"; cc = json.loads(CC.read_text()) if CC.exists() else {}
L0 = open(N/"COMPARACAO_FINAL.md").read().split("\n")
rows = []
for V in sorted(cache):
    split = next((s for s, p in SPLITS.items() if os.path.isdir(f"{p}/{V}")), None)
    if split is None: continue
    r = dict(cache[V]["row"])
    pn = N/"states"/f"{V}.pklz"
    if V in DEC and pn.exists():
        if V in cc: r["nosso"] = cc[V]
        else:
            code, det = g["load"](pn)
            pl = (det.role == "player") & det.track_id.notna()
            novo = det.loc[pl, "track_id"].astype(int).astype(str).map(lambda t: DEC[V].get(t))
            det.loc[pl, "jersey_number"] = novo.map(lambda n: float(n) if n is not None else np.nan)
            j, t, _ = analyse_video(code, det, SPLITS[split])
            r["nosso"] = None if j is None else {"jnr": g["jnr"](j), "team": g["team"](t)}
            cc[V] = r["nosso"]; CC.write_text(json.dumps(cc)); print(V, r["nosso"], flush=True)
    else:
        r["nosso"] = None
    rows.append((V, split, r))
g2 = dict(g); g2["rows"] = rows
corpo = src.split("CACHE.write_text(json.dumps(cache))")[1]
corpo = corpo.replace('(N / "COMPARACAO_FINAL.md")', '(N / "COMPARACAO_FINAL_conf.md")')
corpo = corpo.replace("'Nosso' = nossa pipeline completa (nosso_out).", "'Nosso' = nossa pipeline completa com a decisão pela confiança da leitura (conf_decisoes.json).")
exec(compile(corpo, "compara_final_tail", "exec"), g2)
