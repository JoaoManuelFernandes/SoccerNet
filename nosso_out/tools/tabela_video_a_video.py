"""Tabela vídeo a vídeo (equipa e número, original vs nosso, GS-HOTA por vídeo, veredictos visuais) a partir de
COMPARACAO_FINAL.md, gshota/{original,nosso}_<split>.json e dos ficheiros de veredictos. Saída: TABELA_VIDEO_A_VIDEO.md/.csv"""
import json, re, csv, statistics as st, os
import os as _os; SUF=_os.environ.get("SUF",""); TOUT=_os.environ.get("TESE_OUT","/home/joao/soccernet/tese")
N="/home/joao/soccernet/nosso_out"; B="/home/joao/soccernet/baseline_out"
rows={}
for l in open(f"{N}/COMPARACAO_FINAL{SUF}.md"):
    if l.startswith("| SNGS-"):
        c=[x.strip() for x in l.split("|")[1:-1]]; rows[c[0]]=c
def parse(s):
    m=re.match(r"(\d+)/(\d+) = ([\d.]+)%",s); return (int(m.group(1)),int(m.group(2)),float(m.group(3))) if m else None
g={}
for sp in ("test","valid","train"):
    try: g[sp]=(json.load(open(f"{N}/gshota/original_{sp}.json"))["per_video"],json.load(open(f"{N}/gshota/{'conf' if SUF else 'nosso'}_{sp}.json"))["per_video"])
    except FileNotFoundError: g[sp]=({},{})
def veredictos(path):
    d={}
    for l in open(path):
        if l.startswith("| SNGS-"):
            c=[x.strip() for x in l.split("|")[1:-1]]
            v=c[0]; ver=c[2] if len(c)>=5 and c[1] in ("test","valid","train") else c[1]
            d[v]=ver.split("(")[0].strip() if not ver.startswith("OK (anot") else "OK (anotação invertida)"
    return d
vb=veredictos(f"{B}/VEREDICTOS_VISUAIS_BASELINE.md"); vn=veredictos(f"{N}/VEREDICTOS_VISUAIS_NOSSO.md")
L=["# Vídeo a vídeo: baseline original vs nosso, nas tarefas que alterámos (equipa/lado e número)","",
"Exatidão por tracklet (generate_report.py), mesmos tracklets nos dois sistemas. Δ = nosso − original, em pontos percentuais.",
"GS-HOTA por vídeo (TrackEval) só como referência: cobre todas as tarefas, incluindo as que não tocámos.","",
"| Vídeo | Split | Equipa orig. | Equipa nosso | Δ equipa | Número orig. | Número nosso | Δ número | GS-HOTA orig. | GS-HOTA nosso | Visual baseline | Visual nosso |",
"|---|---|---|---|---|---|---|---|---|---|---|---|"]
csvrows=[]; agg={}
for V,c in sorted(rows.items()):
    sp=c[1]; to,tn,jo,jn=parse(c[2]),parse(c[3]),parse(c[4]),parse(c[6])
    if not (to and tn and jo and jn): continue
    ho=g[sp][0].get(V,{}).get("HOTA",float("nan")); hn=g[sp][1].get(V,{}).get("HOTA",float("nan"))
    L.append(f"| {V[5:]} | {sp} | {to[2]:.1f} | {tn[2]:.1f} | {tn[2]-to[2]:+.1f} | {jo[2]:.1f} | {jn[2]:.1f} | {jn[2]-jo[2]:+.1f} | {ho:.1f} | {hn:.1f} | {vb.get(V,'OK')} | {vn.get(V,'OK')} |")
    csvrows.append([V,sp,to[2],tn[2],jo[2],jn[2],ho,hn,vb.get(V,"OK"),vn.get(V,"OK"),to[0],to[1],tn[0],tn[1],jo[0],jo[1],jn[0],jn[1]])
    a=agg.setdefault(sp,{"to":[0,0],"tn":[0,0],"jo":[0,0],"jn":[0,0],"pto":[],"ptn":[],"pjo":[],"pjn":[]})
    for k,x in (("to",to),("tn",tn),("jo",jo),("jn",jn)): a[k][0]+=x[0]; a[k][1]+=x[1]
    a["pto"].append(to[2]);a["ptn"].append(tn[2]);a["pjo"].append(jo[2]);a["pjn"].append(jn[2])
L+=["","## Médias por conjunto","","| Conjunto | vídeos | Equipa orig. (tracklets) | Equipa nosso (tracklets) | Equipa orig. (média/vídeo) | Equipa nosso (média/vídeo) | Número orig. (tracklets) | Número nosso (tracklets) | Número orig. (média/vídeo) | Número nosso (média/vídeo) |","|---|---|---|---|---|---|---|---|---|---|"]
tot={"to":[0,0],"tn":[0,0],"jo":[0,0],"jn":[0,0],"pto":[],"ptn":[],"pjo":[],"pjn":[]}
f=lambda x:100*x[0]/x[1]
for sp in ("test","valid","train"):
    a=agg[sp]; L.append(f"| {sp} | {len(a['pto'])} | {f(a['to']):.1f} | **{f(a['tn']):.1f}** | {st.mean(a['pto']):.1f} | **{st.mean(a['ptn']):.1f}** | {f(a['jo']):.1f} | **{f(a['jn']):.1f}** | {st.mean(a['pjo']):.1f} | **{st.mean(a['pjn']):.1f}** |")
    for k in ("to","tn","jo","jn"): tot[k][0]+=a[k][0]; tot[k][1]+=a[k][1]
    for k in ("pto","ptn","pjo","pjn"): tot[k]+=a[k]
L.append(f"| **todos** | {len(tot['pto'])} | {f(tot['to']):.1f} | **{f(tot['tn']):.1f}** | {st.mean(tot['pto']):.1f} | **{st.mean(tot['ptn']):.1f}** | {f(tot['jo']):.1f} | **{f(tot['jn']):.1f}** | {st.mean(tot['pjo']):.1f} | **{st.mean(tot['pjn']):.1f}** |")
L+=["","## Quantos vídeos melhoram, empatam ou pioram (Δ por vídeo, tolerância ±1 pp)","","| Conjunto | Equipa: melhora / igual / piora | Número: melhora / igual / piora |","|---|---|---|"]
for sp in ("test","valid","train"):
    a=agg[sp]; dt=[n-o for o,n in zip(a["pto"],a["ptn"])]; dj=[n-o for o,n in zip(a["pjo"],a["pjn"])]
    L.append(f"| {sp} | {sum(d>1 for d in dt)} / {sum(-1<=d<=1 for d in dt)} / {sum(d<-1 for d in dt)} | {sum(d>1 for d in dj)} / {sum(-1<=d<=1 for d in dj)} / {sum(d<-1 for d in dj)} |")
open(f"{N}/TABELA_VIDEO_A_VIDEO{SUF}.md","w").write("\n".join(L)+"\n")
with open(f"{N}/TABELA_VIDEO_A_VIDEO{SUF}.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["video","split","equipa_original","equipa_nosso","numero_original","numero_nosso","gshota_original","gshota_nosso","visual_baseline","visual_nosso","eq_o_ok","eq_o_n","eq_n_ok","eq_n_n","num_o_ok","num_o_n","num_n_ok","num_n_n"]); w.writerows(csvrows)
print("ok", len(csvrows), "vídeos")
