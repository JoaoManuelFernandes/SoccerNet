"""Tabela de tempos por vídeo e por módulo (segundos), pipeline ORIGINAL (a linha [TEMPO] do motor).
Fontes: baseline_out/logs/<V>.log (pipeline: 8 módulos originais + JNR v2 no lugar do OCR) e
baseline_out/logs_ocr/<V>.log (MMOCR original). Total original = soma dos 8 originais + MMOCR.
Saída: baseline_out/TEMPOS_POR_VIDEO.md"""
import re, glob, os, statistics as st
OUT="/home/joao/soccernet/baseline_out"
MODS=["YOLOv8","PRTReId","BPBReIDStrongSORT","NBJW_Calib_Keypoints","NBJW_Calib","MMOCR","TrackletsJNRv2","MajorityVoteTracklet","TrackletTeamClustering","TrackletTeamSideLabeling"]
NOMES={"YOLOv8":"deteção","PRTReId":"ReID","BPBReIDStrongSORT":"seguimento","NBJW_Calib_Keypoints":"campo","NBJW_Calib":"calibração","MMOCR":"OCR original","TrackletsJNRv2":"JNR v2 (nosso)","MajorityVoteTracklet":"voto tracklet","TrackletTeamClustering":"equipa","TrackletTeamSideLabeling":"lado"}
def tempos(f):
    d={}
    if os.path.exists(f):
        for m in re.finditer(r"^\[TEMPO\] (\S+) ([\d.]+)s", open(f,errors="ignore").read(), re.M): d[m.group(1)]=float(m.group(2))
    return d
rows=[]
for f in sorted(glob.glob(f"{OUT}/logs/SNGS-*.log")):
    V=os.path.basename(f)[:-4]; t=tempos(f); t.update({k:v for k,v in tempos(f"{OUT}/logs_ocr/{V}.log").items() if k=="MMOCR"})
    if not any(k in t for k in MODS): continue
    rows.append((V,t))
L=["# Tempos por vídeo e por módulo, pipeline original (segundos)","",
   "Cronómetro do motor (`[TEMPO]`), GPU RTX 3060 portátil 6 GB, 3 workers de dados. Só a partir de 2026-09-13 20:45 (SNGS-130) e das passagens OCR.",
   "Sem cronómetro: carregamento do vídeo e render do MP4 (~3 min no total por vídeo).",
   "**Total original** = deteção + ReID + seguimento + campo + calibração + OCR original + voto + equipa + lado (sem o JNR v2).","",
   "| Vídeo | "+" | ".join(NOMES[m] for m in MODS)+" | total original |","|---|"+"---|"*(len(MODS)+1)]
tot={m:[] for m in MODS}; totals=[]
for V,t in rows:
    cells=[]
    for m in MODS:
        v=t.get(m); cells.append(f"{v:.1f}" if v is not None else "—")
        if v is not None: tot[m].append(v)
    orig=[t.get(m) for m in MODS if m!="TrackletsJNRv2"]
    tt=sum(orig) if all(v is not None for v in orig) else None
    if tt is not None: totals.append(tt)
    L.append(f"| {V} | "+" | ".join(cells)+f" | {tt/60:.1f} min |" if tt is not None else f"| {V} | "+" | ".join(cells)+" | — |")
L+=["", "| mediana | "+" | ".join(f"{st.median(tot[m]):.1f} (n={len(tot[m])})" if tot[m] else "—" for m in MODS)+(f" | {st.median(totals)/60:.1f} min (n={len(totals)}) |" if totals else " | — |")]
open(f"{OUT}/TEMPOS_POR_VIDEO.md","w").write("\n".join(L)+"\n"); print(L[-1])
