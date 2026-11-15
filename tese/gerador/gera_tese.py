"""Gera tese/Chapter4_final.tex a partir do template e dos ficheiros de dados (nosso_out). Reexecutar após atualizar os dados:
  venv_metrics/bin/python tese/gerador/gera_cap4.py"""
import csv, json, re, statistics as st, os
import os as _os; SUF=_os.environ.get("SUF",""); TOUT=_os.environ.get("TESE_OUT","/home/joao/soccernet/tese")
TPL=_os.environ.get("TPL_DIR","/home/joao/soccernet/tese/gerador")
N="/home/joao/soccernet/nosso_out"; T="/home/joao/soccernet/tese"
rows=list(csv.DictReader(open(f"{N}/TABELA_VIDEO_A_VIDEO{SUF}.csv")))
tok={}
def pct(a,b): return f"${100*a/b:.1f}\\%$"
def frac(a,b): return f"${a}/{b}$"
def dpp(o,n): return f"${n-o:+.1f}$"
# --- exatidão por conjunto (agregada por tracklet)
for k,pre in (("eq","TEAM"),("num","JNR")):
    tot=[0,0,0,0]
    for sp,name in (("test","TEST"),("valid","VALID"),("train","TRAIN")):
        rs=[r for r in rows if r["split"]==sp]
        oo,on,no,nn=(sum(int(r[f"{k}_o_ok"]) for r in rs),sum(int(r[f"{k}_o_n"]) for r in rs),sum(int(r[f"{k}_n_ok"]) for r in rs),sum(int(r[f"{k}_n_n"]) for r in rs))
        tok[f"{pre}_{name}_O"]=pct(oo,on); tok[f"{pre}_{name}_OF"]=frac(oo,on); tok[f"{pre}_{name}_N"]=pct(no,nn); tok[f"{pre}_{name}_NF"]=frac(no,nn); tok[f"{pre}_{name}_D"]=dpp(100*oo/on,100*no/nn)
        for i,v in enumerate((oo,on,no,nn)): tot[i]+=v
    oo,on,no,nn=tot
    tok[f"{pre}_ALL_O"]=pct(oo,on); tok[f"{pre}_ALL_OF"]=frac(oo,on); tok[f"{pre}_ALL_N"]=pct(no,nn); tok[f"{pre}_ALL_NF"]=frac(no,nn); tok[f"{pre}_ALL_D"]=dpp(100*oo/on,100*no/nn)
# JNR v2 na pipeline original (coluna 6 de COMPARACAO_FINAL.md)
for l in open(f"{N}/COMPARACAO_FINAL{SUF}.md"):
    if l.startswith("| test") or l.startswith("| valid") or l.startswith("| train"):
        c=[x.strip() for x in l.split("|")[1:-1]]; m=re.search(r"= ([\d.]+)%",c[5]); tok[f"JNR_{c[0].upper()}_N2"]=f"${m.group(1)}\\%$"
# --- distribuições por vídeo
dj=[float(r["numero_nosso"])-float(r["numero_original"]) for r in rows]; dt=[float(r["equipa_nosso"])-float(r["equipa_original"]) for r in rows]
tok["JNR_UP"]=f"${sum(d>1 for d in dj)}$"; tok["JNR_MED"]=f"${st.median(dj):.1f}$"; tok["JNR_GE50"]=f"${sum(float(r['numero_nosso'])>=50 for r in rows)}$"
for nm,lo,hi in (("JB40",40,999),("JB30",30,40),("JB20",20,30),("JB10",10,20),("JB1",1,10)): tok[nm]=f"${sum(1 for d in dj if lo<d<=hi)}$"
tok["JB0"]=f"${sum(1 for d in dj if -1<=d<=1)}$"; tok["JBN"]=f"${sum(1 for d in dj if d<-1)}$"
tok["TEAM_UP"]=f"${sum(d>1 for d in dt)}$"; tok["TEAM_UP95"]=f"${sum(1 for r in rows if float(r['equipa_nosso'])-float(r['equipa_original'])>1 and float(r['equipa_nosso'])>=95)}$"; tok["TEAM_EQ"]=f"${sum(-1<=d<=1 for d in dt)}$"; tok["TEAM_DN"]=f"${sum(d<-1 for d in dt)}$"
# --- categorias dos vídeos em que a métrica dá o proposto abaixo da referência
COLAPSO={"SNGS-025","SNGS-027","SNGS-028","SNGS-030","SNGS-036","SNGS-038","SNGS-069","SNGS-091","SNGS-131"}; LADO={"SNGS-113","SNGS-106"}
AMARELO={f"SNGS-{n:03d}" for n in range(39,59)}
dn=[r["video"] for r in rows if float(r["equipa_nosso"])-float(r["equipa_original"])<-1]
tok["DN_COLAPSO"]=f"${sum(v in COLAPSO for v in dn)}$"; tok["DN_LADO"]=f"${sum(v in LADO for v in dn)}$"; tok["DN_AMARELO"]=f"${sum(v in AMARELO for v in dn)}$"
tok["DN_OUTROS"]=f"${sum(v not in COLAPSO|LADO|AMARELO for v in dn)}$"
# --- inspeção visual (contagens fixadas a partir dos ficheiros de veredictos)
# Veredictos visuais. Os do sistema proposto foram atualizados a 2026-09-22 para os dez vídeos
# reprocessados pela verificação de partições degeneradas: sete passaram de agrupamento colapsado a
# correto (exatidão de 100% contra a anotação) e o SNGS-131 passou a agrupar bem, ficando na
# categoria de anotação invertida.
vis={"O":dict(OK=111,LADO=7,MESMA=7,MIST=14,FUNC=20,ANOT=5),"N":dict(OK=132,LADO=3,MESMA=1,MIST=3,FUNC=20,ANOT=4)}
for _s,_d in vis.items():
    _tot=_d["OK"]+_d["LADO"]+_d["MESMA"]+_d["MIST"]+_d["FUNC"]
    assert _tot==159, f"contagens visuais de {_s} somam {_tot}, deviam somar 159 (ANOT conta dentro de OK)"
for s,d in vis.items():
    for k,v in d.items(): tok[f"VIS_{s}_{k}"]=str(v)
    tok[f"VIS_{s}_EQUIPA"]=str(d["LADO"]+d["MESMA"]+d["MIST"])
tok["N_ANOT"]="cinco"
# exatidão da equipa excluindo os cinco vídeos com a anotação do lado invertida
INVERT={"SNGS-126","SNGS-131","SNGS-197","SNGS-092","SNGS-111"}
cl=[r for r in rows if r["video"] not in INVERT]
for pre,k in (("TEAM","eq"),("JNR","num")):
    oo=sum(int(r[f"{k}_o_ok"]) for r in cl); on=sum(int(r[f"{k}_o_n"]) for r in cl)
    no=sum(int(r[f"{k}_n_ok"]) for r in cl); nn=sum(int(r[f"{k}_n_n"]) for r in cl)
    tok[f"{pre}_CLEAN_O"]=pct(oo,on); tok[f"{pre}_CLEAN_N"]=pct(no,nn); tok[f"{pre}_CLEAN_D"]=dpp(100*oo/on,100*no/nn)
# --- tabela de regressões (ramo e grupos do registo de execução; exatidões da tabela)
reg=[("SNGS-025","cor (concentração)","64/15"),("SNGS-027","cor (concentração)","17/60"),("SNGS-030","cor (concentração)","38/7"),("SNGS-038","cor (concentração)","5/39"),("SNGS-131","cor (concentração)","54/15"),("SNGS-069","cor (silhueta)","11/39"),("SNGS-028","reidentificação","9/44"),("SNGS-036","reidentificação","41/2"),("SNGS-091","reidentificação","10/60")]
byv={r["video"]:r for r in rows}
# (a tabela de partições degeneradas passou a ser fixa no template, com os valores antes/depois da auditoria)
# --- GS-HOTA
gs={}
for l in open(f"{N}/gshota/RESUMO_GSHOTA{SUF}.md"):
    m=re.match(r"\| (original|nosso) \| (test|valid|train) \| \d+ \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|",l)
    if m: gs[(m.group(1),m.group(2))]=[float(m.group(i)) for i in (3,4,5,6)]
NOME={"test":"Teste","valid":"Validação","train":"Treino"}
lines=[]
for sp in ("test","valid","train"):
    for sysn,lab in (("original","Referência"),("nosso","Proposto")):
        h,d,a,lo=gs[(sysn,sp)]; lines.append(f"        {NOME[sp] if sysn=='original' else ''} & {lab} & ${h:.2f}\\%$ & ${d:.1f}$ & ${a:.1f}$ & ${lo:.1f}$ \\\\")
    tok[f"GS_{sp.upper()}_O"]=f"${gs[('original',sp)][0]:.2f}\\%$"; tok[f"GS_{sp.upper()}_N"]=f"${gs[('nosso',sp)][0]:.2f}\\%$"; tok[f"GS_{sp.upper()}_D"]=f"${gs[('nosso',sp)][0]-gs[('original',sp)][0]:.1f}$"
tok["TAB_GSHOTA"]="\n        \\hline\n".join(lines)
# por vídeo
piora=[]; 
for sp in ("test","valid","train"):
    dd=[float(r["gshota_nosso"])-float(r["gshota_original"]) for r in rows if r["split"]==sp]
    tok[f"GS_MED_{sp.upper()}"]=f"${st.median(dd):+.1f}$"
    piora+=[r["video"][5:] for r in rows if r["split"]==sp and float(r["gshota_nosso"])-float(r["gshota_original"])<-1]
tok["GS_PIORA"]=(f"{len(piora)} (SNGS-"+", SNGS-".join(piora)+")") if piora else "nenhum"
# --- decomposição
dec={}
for l in open(f"{N}/gshota/DECOMPOSICAO_ATRIBUTOS{SUF}.md"):
    m=re.match(r"\| (original|nosso) \| (test|valid|train) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|",l)
    if m: dec[(m.group(1),m.group(2))]=[float(m.group(i)) for i in (3,4,5,6)]
lines=[]
for sp in ("test","valid","train"):
    for sysn,lab in (("original","Referência"),("nosso","Proposto")):
        y=dec[(sysn,sp)]; lines.append(f"        {NOME[sp] if sysn=='original' else ''} & {lab} & ${y[0]:.1f}$ & ${y[1]:.1f}$ & ${y[2]:.1f}$ & ${y[3]:.1f}$ \\\\")
tok["TAB_DECOMP"]="\n        \\hline\n".join(lines)
pos=[dec[("nosso",sp)][0] for sp in ("test","valid","train")]; tok["POS_MIN"]=f"${min(pos):.1f}\\%$"; tok["POS_MAX"]=f"${max(pos):.1f}\\%$"; tok["POS_TEST_N"]=f"${dec[('nosso','test')][0]:.0f}$"
for sp in ("test","valid","train"):
    o,n=dec[("original",sp)],dec[("nosso",sp)]
    tok[f"FUNC_{sp.upper()}_O"]=f"${o[0]-o[1]:.1f}$"; tok[f"EQ_{sp.upper()}_O"]=f"${o[1]-o[2]:.1f}$"; tok[f"NUM_{sp.upper()}_O"]=f"${o[2]-o[3]:.1f}$"
    tok[f"EQ_{sp.upper()}_N"]=f"${n[1]-n[2]:.1f}$"; tok[f"NUM_{sp.upper()}_N"]=f"${n[2]-n[3]:.1f}$"
    tok[f"GANHO_EQ_{sp.upper()}"]=f"${(o[1]-o[2])-(n[1]-n[2]):.1f}$"; tok[f"GANHO_NUM_{sp.upper()}"]=f"${(o[2]-o[3])-(n[2]-n[3]):.1f}$"
# defeito de calibração = referência corrigida menos referência na configuração por omissão (batch 8), no teste:
# em "só posição" (onde o defeito atua diretamente) e na métrica completa
b8={}
for l in open(f"{N}/gshota/DECOMPOSICAO_ATRIBUTOS_original_batch8.md"):
    m=re.match(r"\| (original) \| (test) \| ([\d.]+) \|",l)
    if m: b8["pos"]=float(m.group(3))
for l in open(f"{N}/gshota/RESUMO_GSHOTA_original_batch8.md"):
    m=re.match(r"\| original \| test \| \d+ \| ([\d.]+) \|",l)
    if m: b8["full"]=float(m.group(1))
tok["CALIB_GANHO"]=f"${dec[('original','test')][0]-b8['pos']:.0f}$"
tok["CALIB_GANHO_FULL"]=f"${gs[('original','test')][0]-b8['full']:.1f}$"
# "só posição" tem de coincidir nos dois sistemas (mesmas deteções, mesmos tracks, mesma calibração)
pdiff=max(abs(dec[("nosso",sp)][0]-dec[("original",sp)][0]) for sp in ("test","valid","train"))
tok["POS_COINCIDEM"]="coincidem, como têm de coincidir," if pdiff<0.05 else f"coincidem a menos de ${pdiff:.1f}$ pontos, como têm de coincidir,"
# atributo que mais custa, por sistema e conjunto (função, equipa, número); exceções à regra "é o número"
NOMEA={"f":"a função","e":"a equipa","n":"o número"}; SYSN={"original":"solução de referência","nosso":"sistema proposto"}
exc=[]
for sysn in ("original","nosso"):
    for sp in ("test","valid","train"):
        y=dec[(sysn,sp)]; c={"f":y[0]-y[1],"e":y[1]-y[2],"n":y[2]-y[3]}; k=max(c,key=c.get)
        if k!="n":
            extra=", pelos vinte vídeos da equipa de amarelo," if (sp=="valid" and k=="f") else ""
            exc.append(f"{'na' if sp!='test' else 'no'} {NOME[sp].lower()} {'do' if sysn=='nosso' else 'da'} {SYSN[sysn]}, onde {NOMEA[k]}{extra} custa mais do que o número (${c[k]:.1f}$ contra ${c['n']:.1f}$ pontos)")
tok["NUM_EXCECOES"]="" if not exc else (", com uma única exceção, " if len(exc)==1 else ", com as exceções seguintes: ")+" e ".join(exc)
# atribuição do ganho por conjunto: equipa + número (+ resto, quando não desprezável)
for sp in ("test","valid","train"):
    o,n=dec[("original",sp)],dec[("nosso",sp)]; ge=(o[1]-o[2])-(n[1]-n[2]); gn=(o[2]-o[3])-(n[2]-n[3]); tot=n[3]-o[3]; resto=tot-ge-gn
    pe=f"${ge:.1f}$ vêm da equipa" if ge>=0.05 else (f"a equipa não contribui (${ge:.1f}$)" if ge<-0.05 else "a equipa não contribui")
    pn=f"${gn:.1f}$ do número" if ge>=0.05 else f"${gn:.1f}$ vêm do número"
    rpos=n[0]-o[0]
    pr="" if abs(resto)<0.15 else (f" (os restantes ${resto:.1f}$ são a diferença no custo da função)" if abs(rpos)<0.05 else f" (os restantes ${resto:.1f}$ vêm da posição e da função)")
    tok[f"ATRIB_{sp.upper()}"]=f"{pe} e {pn}{pr}"
# --- substituir
for tpl,out in (("Chapter4_template.tex","Chapter4_final.tex"),("Conclusao_template.tex","Chapter5_Conclusao_final.tex"),("Resumo_template.tex","Resumo_final.tex"),("Abstract_template.tex","Abstract_final.tex"),("Introducao_template.tex","Chapter1_Introducao_final.tex")):
    s=open(f"{TPL}/{tpl}").read()
    falta=set(re.findall(r"@@([A-Z0-9_]+)@@",s))-set(tok)
    for k,v in tok.items(): s=s.replace(f"@@{k}@@",v)
    open(f"{TOUT}/{out}","w").write(s)
    print(f"{out} gerado;", "tokens em falta:", sorted(falta) if falta else "nenhum")
