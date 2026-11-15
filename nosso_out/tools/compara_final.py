"""Tabela final por vídeo: baseline ORIGINAL vs NOSSO desenvolvimento, nos 160 vídeos.
Colunas: equipa original (states_ocr = pipeline 100% original), equipa nossa (nosso_out/states),
número OCR original (states_ocr), número JNR v2 na pipeline original (baseline_out/states),
número JNR v2 na nossa pipeline (nosso_out/states). Métrica: analyse_video de generate_report.py.
Cache incremental em compara_final_cache.json. Saída: nosso_out/COMPARACAO_FINAL.md"""
import sys, zipfile, pickle, glob, os, json, re
from pathlib import Path
sys.path.insert(0, "/home/joao/soccernet/sn-gamestate")
from generate_report import analyse_video
B = Path("/home/joao/soccernet/baseline_out"); N = Path("/home/joao/soccernet/nosso_out")
SPLITS = {s: f"/home/joao/soccernet/data/SoccerNetGS/{s}" for s in ("test", "valid", "train")}
CACHE = N / "tools" / "compara_final_cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

def load(p):
    with zipfile.ZipFile(p) as zf:
        code = [n for n in zf.namelist() if n.endswith(".pkl") and not n.endswith("_image.pkl")][0][:-4]
        with zf.open(f"{code}.pkl") as f: return code, pickle.load(f)
def jnr(df):
    e = df[df.eval_status != "skipped"]; c = e[e.eval_status != "id_switch"]
    return int((c.eval_status == "correct").sum()), int(len(c))
def team(df):
    e = df[df.eval_status != "skipped"]; c = e[e.eval_status != "id_switch"]
    return int((c.eval_status == "correct").sum()), int(len(c))
def key(*paths): return "|".join(f"{p.stat().st_size}:{int(p.stat().st_mtime)}" if p.exists() else "0" for p in paths)
def eval_state(p, gt):
    try: code, det = load(p)
    except Exception: return None
    j, t, _ = analyse_video(code, det, gt)
    if j is None: return None
    return {"jnr": jnr(j), "team": team(t)}

rows = []
allv = sorted({os.path.basename(p)[:-5] for d in (B/"states", B/"states_ocr", N/"states") for p in glob.glob(str(d/"SNGS-*.pklz"))})
for V in allv:
    split = next((s for s, p in SPLITS.items() if os.path.isdir(f"{p}/{V}")), None)
    if split is None: continue
    # "original" = estados 100% originais recalibrados frame a frame (states_ocr_c1) quando existem; senão states_ocr (batch 8)
    po = B/"states_ocr_c1"/f"{V}.pklz" if (B/"states_ocr_c1"/f"{V}.pklz").exists() else B/"states_ocr"/f"{V}.pklz"
    pb, pn = B/"states"/f"{V}.pklz", N/"states"/f"{V}.pklz"
    k = key(po, pb, pn)
    if V in cache and cache[V]["key"] == k: r = cache[V]["row"]
    else:
        r = {"orig": eval_state(po, SPLITS[split]) if po.exists() else None,
             "base_jnrv2": eval_state(pb, SPLITS[split]) if pb.exists() else None,
             "nosso": eval_state(pn, SPLITS[split]) if pn.exists() else None}
        cache[V] = {"key": k, "row": r}
    rows.append((V, split, r))
CACHE.write_text(json.dumps(cache))

def pct(x): return f"{x[0]}/{x[1]} = {100*x[0]/x[1]:.1f}%" if x and x[1] else "—"
L = ["# Baseline ORIGINAL vs NOSSO desenvolvimento, por vídeo (160)", "",
     "Equipa e número por tracklet (generate_report.py). 'Original' = pipeline 100% original (states_ocr). 'Nosso' = nossa pipeline completa (nosso_out).",
     "'JNR v2 na pipeline original' = o nosso leitor sobre os tracklets originais (mesmos tracklets que o OCR original).",
     "Veredictos visuais: baseline em baseline_out/VEREDICTOS_VISUAIS_BASELINE.md; nosso em docs/FALHAS_VALIDADAS_VISUALMENTE.md.", "",
     "| Vídeo | Split | Equipa original | Equipa nossa | Nº OCR original | Nº JNR v2 (pipeline original) | Nº JNR v2 (nossa pipeline) |", "|---|---|---|---|---|---|---|"]
tot = {}
for V, split, r in rows:
    o, b, n = r["orig"], r["base_jnrv2"], r["nosso"]
    L.append(f"| {V} | {split} | {pct(o and o['team'])} | {pct(n and n['team'])} | {pct(o and o['jnr'])} | {pct(b and b['jnr'])} | {pct(n and n['jnr'])} |")
    t = tot.setdefault(split, {"to": [0, 0], "tn": [0, 0], "jo": [0, 0], "jb": [0, 0], "jn": [0, 0], "n": 0, "nn": 0})
    t["n"] += 1
    if o: t["to"][0] += o["team"][0]; t["to"][1] += o["team"][1]; t["jo"][0] += o["jnr"][0]; t["jo"][1] += o["jnr"][1]
    if b: t["jb"][0] += b["jnr"][0]; t["jb"][1] += b["jnr"][1]
    if n: t["nn"] += 1; t["tn"][0] += n["team"][0]; t["tn"][1] += n["team"][1]; t["jn"][0] += n["jnr"][0]; t["jn"][1] += n["jnr"][1]
L += ["", "## Totais por conjunto", "", "| Conjunto | vídeos (com nosso) | Equipa original | Equipa nossa | Nº OCR original | Nº JNR v2 (pipeline original) | Nº JNR v2 (nossa pipeline) |", "|---|---|---|---|---|---|---|"]
for s in ("test", "valid", "train"):
    if s in tot:
        t = tot[s]; L.append(f"| {s} | {t['n']} ({t['nn']}) | {pct(t['to'])} | {pct(t['tn'])} | {pct(t['jo'])} | {pct(t['jb'])} | {pct(t['jn'])} |")
(N / "COMPARACAO_FINAL.md").write_text("\n".join(L) + "\n")
print(L[-3:])
