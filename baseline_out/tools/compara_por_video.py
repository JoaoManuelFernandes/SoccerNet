"""Tabela por vídeo: número (OCR original vs JNR v2, mesmos tracklets) e equipa (módulos originais).
Fontes: baseline_out/states/<V>.pklz (pipeline original + JNR v2) e baseline_out/states_ocr/<V>.pklz
(idem, com o MMOCR original). Avaliação: analyse_video de sn-gamestate/generate_report.py.
Saída: baseline_out/COMPARACAO_POR_VIDEO.md (reescrita a cada execução)."""
import sys, zipfile, pickle, glob, os
from pathlib import Path
sys.path.insert(0, "/home/joao/soccernet/sn-gamestate")
from generate_report import analyse_video
OUT = Path("/home/joao/soccernet/baseline_out")
SPLITS = {"test": "/home/joao/soccernet/data/SoccerNetGS/test", "valid": "/home/joao/soccernet/data/SoccerNetGS/valid", "train": "/home/joao/soccernet/data/SoccerNetGS/train"}

def load(pklz):
    with zipfile.ZipFile(pklz) as zf:
        code = [n for n in zf.namelist() if n.endswith(".pkl") and not n.endswith("_image.pkl")][0][:-4]
        with zf.open(f"{code}.pkl") as f: return code, pickle.load(f)

def jnr_stats(jnr_df):
    e = jnr_df[jnr_df.eval_status != "skipped"]; c = e[e.eval_status != "id_switch"]
    ok, wr, no = (c.eval_status == "correct").sum(), (c.eval_status == "wrong").sum(), (c.eval_status == "no_prediction").sum()
    return ok, wr, no, len(c)

def team_stats(team_df):
    e = team_df[team_df.eval_status != "skipped"]; c = e[e.eval_status != "id_switch"]
    return (c.eval_status == "correct").sum(), len(c)

import json
CACHE = OUT / "tools" / "compara_cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
rows = []
for st in sorted(glob.glob(str(OUT / "states" / "SNGS-*.pklz"))):
    V = os.path.basename(st)[:-5]
    split = next((s for s, p in SPLITS.items() if os.path.isdir(f"{p}/{V}")), None)
    if split is None: continue
    st_ocr = OUT / "states_ocr" / f"{V}.pklz"
    has_ocr = st_ocr.exists() and st_ocr.stat().st_size > 0
    key = f"{os.path.getsize(st)}|{int(os.path.getmtime(st))}|{st_ocr.stat().st_size if has_ocr else 0}"
    if V in cache and cache[V]["key"] == key:
        r = cache[V]["row"]
    else:
        try:
            code, det = load(st)
        except Exception:
            continue   # estado a ser escrito neste momento
        jnr_v2, team_df, _ = analyse_video(code, det, SPLITS[split])
        if jnr_v2 is None: continue
        ok2, wr2, no2, n2 = jnr_stats(jnr_v2); tok, tn = team_stats(team_df)
        if has_ocr:
            try:
                _, det_o = load(st_ocr); jnr_o, _, _ = analyse_video(code, det_o, SPLITS[split]); ok1, wr1, no1, n1 = jnr_stats(jnr_o)
            except Exception:
                ok1 = wr1 = no1 = n1 = None
        else: ok1 = wr1 = no1 = n1 = None
        r = [V, split, ok1, wr1, no1, n1, int(ok2), int(wr2), int(no2), int(n2), int(tok), int(tn)]
        r = [int(x) if hasattr(x, "item") else x for x in r]
        cache[V] = {"key": key, "row": r}
    rows.append(tuple(r) + ((OUT / "videos" / f"{V}.mp4").exists(),))
CACHE.write_text(json.dumps(cache))

def pct(a, b): return f"{100*a/b:.1f}%" if b else "—"
L = ["# Comparação por vídeo: baseline original vs nosso desenvolvimento", "",
     "Mesmas deteções, mesmo seguimento e mesma calibração em todas as colunas (estados da corrida da baseline).",
     "**Número**: OCR original = MMOCR (DBNet + SAR), o código original; JNR v2 = o nosso módulo (Qwen2.5-VL-3B).",
     "**Equipa (original)**: módulos `team`/`team_side` originais (k-means + comparação de médias), métrica por tracklet contra a anotação.",
     "A coluna do nosso team_side virá dos nossos estados; o veredicto visual do João está em docs/FALHAS_VALIDADAS_VISUALMENTE.md.",
     "Filtros da avaliação: os de generate_report.py (min_frames, match_ratio, IoU, pureza).", "",
     "| Vídeo | Split | OCR original: certos/errados/sem leitura | Exatidão | JNR v2: certos/errados/sem leitura | Exatidão | Tracklets c/ número | Equipa original | Vídeo renderizado |",
     "|---|---|---|---|---|---|---|---|---|"]
tot = dict(ok1=0, n1=0, ok2=0, n2=0, tok=0, tn=0)
for V, split, ok1, wr1, no1, n1, ok2, wr2, no2, n2, tok, tn, vid in rows:
    o = f"{ok1}/{wr1}/{no1}" if n1 is not None else "(por correr)"
    L.append(f"| {V} | {split} | {o} | {pct(ok1, n1) if n1 else '—'} | {ok2}/{wr2}/{no2} | {pct(ok2, n2)} | {n2} | {tok}/{tn} = {pct(tok, tn)} | {'sim' if vid else 'não'} |")
    if n1: tot["ok1"] += ok1; tot["n1"] += n1
    tot["ok2"] += ok2; tot["n2"] += n2; tot["tok"] += tok; tot["tn"] += tn
L += ["", f"**Totais** ({len(rows)} vídeos com estado; {sum(1 for r in rows if r[5])} com OCR original): "
      f"OCR original {tot['ok1']}/{tot['n1']} = {pct(tot['ok1'], tot['n1'])}; "
      f"JNR v2 {tot['ok2']}/{tot['n2']} = {pct(tot['ok2'], tot['n2'])} (nos mesmos {tot['n1']} tracklets quando o OCR existe, ver tabela); "
      f"equipa original {tot['tok']}/{tot['tn']} = {pct(tot['tok'], tot['tn'])}."]
(OUT / "COMPARACAO_POR_VIDEO.md").write_text("\n".join(L) + "\n")
print("\n".join(L[-1:])); print(f"escrito: {OUT/'COMPARACAO_POR_VIDEO.md'} ({len(rows)} vídeos)")

# tabela de tempos por vídeo (mesma cadência)
import subprocess, sys as _s
subprocess.run([_s.executable, "/home/joao/soccernet/baseline_out/tools/tempos_por_video.py"], check=False)
