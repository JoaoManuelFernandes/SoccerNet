"""Anexo: tabela vídeo a vídeo (159 vídeos) em LaTeX (longtable). Lê nosso_out/TABELA_VIDEO_A_VIDEO.csv. Saída: tese/Anexo_final.tex"""
import csv
import os as _os; SUF=_os.environ.get("SUF",""); TOUT=_os.environ.get("TESE_OUT","/home/joao/soccernet/tese")
N="/home/joao/soccernet/nosso_out"; T="/home/joao/soccernet/tese"
rows=list(csv.DictReader(open(f"{N}/TABELA_VIDEO_A_VIDEO{SUF}.csv")))
NOME={"test":"teste","valid":"validação","train":"treino"}
def ab(v):
    v=v.strip()
    if v.startswith("OK (anot"): return "OK (anot.\\ inv.)"
    if v.startswith("OK"): return "OK"
    if "lado" in v: return "lado"
    if "mesma" in v or "colaps" in v: return "colapsado"
    if "mist" in v: return "misturado"
    if "função" in v or "funcao" in v: return "função"
    return v.replace("%","\\%")
L=[r"\chapter{Resultados por Vídeo}\label{anexo:videos}",r"\setcounter{table}{0}\renewcommand{\thetable}{A.\arabic{table}}","",
r"A Tabela~\ref{tab:anexo-videos} apresenta, para cada um dos $159$ vídeos, a exatidão por \emph{tracklet} da equipa e lado do campo e do número de camisola nos dois sistemas, o GS-HOTA por vídeo e o veredicto da inspeção visual, que classifica cada sequência como correta, com o lado trocado, com o agrupamento colapsado, com o agrupamento misturado, com uma equipa classificada como arbitragem, ou como correta com a anotação de referência invertida.","",
r"\begin{small}",r"\begin{longtable}{|l|l|r|r|r|r|r|r|l|l|}",
r"\caption{Resultados por vídeo. Ref.\ é a solução de referência corrigida e Prop.\ o sistema proposto.}\label{tab:anexo-videos}\\",
r"\hline \textbf{Vídeo} & \textbf{Conj.} & \multicolumn{2}{|c|}{\textbf{Equipa (\%)}} & \multicolumn{2}{|c|}{\textbf{Número (\%)}} & \multicolumn{2}{|c|}{\textbf{GS-HOTA (\%)}} & \multicolumn{2}{|c|}{\textbf{Visual}} \\",
r" & & Ref. & Prop. & Ref. & Prop. & Ref. & Prop. & Ref. & Prop. \\ \hline",r"\endfirsthead",
r"\hline \textbf{Vídeo} & \textbf{Conj.} & Ref. & Prop. & Ref. & Prop. & Ref. & Prop. & Ref. & Prop. \\ \hline",r"\endhead",r"\hline",r"\endfoot"]
for r in sorted(rows,key=lambda r:(("test","valid","train").index(r["split"]),r["video"])):
    L.append(f"{r['video']} & {NOME[r['split']]} & {float(r['equipa_original']):.1f} & {float(r['equipa_nosso']):.1f} & {float(r['numero_original']):.1f} & {float(r['numero_nosso']):.1f} & {float(r['gshota_original']):.1f} & {float(r['gshota_nosso']):.1f} & {ab(r['visual_baseline'])} & {ab(r['visual_nosso'])} \\\\ \\hline")
L+=[r"\end{longtable}",r"\end{small}"]
open(f"{TOUT}/Anexo_final.tex","w").write("\n".join(L)+"\n"); print("Anexo_final.tex gerado:",len(rows),"vídeos")
