"""Figuras do Capítulo 4 (Resultados). Lê nosso_out/TABELA_VIDEO_A_VIDEO.csv e nosso_out/gshota/DECOMPOSICAO_ATRIBUTOS.md.
Gera: fig_delta_por_video.pdf, fig_decomposicao_gshota.pdf, fig_frames_148.pdf (PDF+PNG). Reexecutar após atualizar os dados."""
import csv, re, subprocess, os
import os as _os; SUF=_os.environ.get("SUF",""); TOUT=_os.environ.get("TESE_OUT","/home/joao/soccernet/tese")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
NOSSO="/home/joao/soccernet/nosso_out"; OUT=os.environ.get("FIG_OUT",os.path.dirname(os.path.abspath(__file__))); TESE="/home/joao/soccernet/tese"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#777","axes.linewidth":0.6,"xtick.color":"#444","ytick.color":"#444","axes.labelcolor":"#222"})
COR={"test":"#0072B2","valid":"#E69F00","train":"#009E73"}; NOME={"test":"teste","valid":"validação","train":"treino"}
rows=list(csv.DictReader(open(f"{NOSSO}/TABELA_VIDEO_A_VIDEO{SUF}.csv")))

# 1) Δ por vídeo, número e equipa
fig,axs=plt.subplots(2,1,figsize=(7.2,4.6),sharex=False)
for ax,(k,tit) in zip(axs,[("numero","Número de camisola"),("equipa","Equipa e lado do campo")]):
    d=sorted(((float(r[f"{k}_nosso"])-float(r[f"{k}_original"]),r["split"],r["video"]) for r in rows),key=lambda t:t[0])
    xs=range(len(d)); ax.axhline(0,color="#999",lw=0.6)
    ax.bar(xs,[t[0] for t in d],width=1.0,color=[COR[t[1]] for t in d],edgecolor="none")
    piora=[t for t in d if t[0]<-1]
    if piora: ax.text(0.01,0.86,"pioram: "+", ".join(t[2][5:] for t in piora),transform=ax.transAxes,va="top",fontsize=7.2,color="#555")
    ax.set_ylabel("Δ exatidão (pp)"); ax.set_title(tit,loc="left",fontsize=9.5,fontweight="bold"); ax.set_xlim(-1,len(d)); ax.set_xticks([])
    n_up=sum(1 for t in d if t[0]>1); n_eq=sum(1 for t in d if -1<=t[0]<=1); n_dn=len(d)-n_up-n_eq
    ax.text(0.01,0.97,f"melhoram {n_up} · empatam {n_eq} · pioram {n_dn}  (de {len(d)} vídeos)",transform=ax.transAxes,va="top",fontsize=8,color="#333")
axs[1].set_xlabel("vídeos ordenados pelo ganho (proposto − referência)")
axs[0].legend(handles=[Patch(color=COR[s],label=NOME[s]) for s in ("test","valid","train")],loc="upper left",bbox_to_anchor=(0.0,0.86),frameon=False,fontsize=8,ncol=3)
fig.tight_layout(h_pad=1.2); fig.savefig(f"{OUT}/fig_delta_por_video.pdf"); fig.savefig(f"{OUT}/fig_delta_por_video.png",dpi=200); plt.close(fig)

# 2) decomposição do GS-HOTA por atributo
dec={}
for l in open(f"{NOSSO}/gshota/DECOMPOSICAO_ATRIBUTOS{SUF}.md"):
    m=re.match(r"\| (original|nosso) \| (test|valid|train) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|",l)
    if m: dec[(m.group(1),m.group(2))]=[float(m.group(i)) for i in (3,4,5,6)]
niv=["posição","+ função","+ equipa","+ número"]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.7),sharey=True)
for ax,sp in zip(axs,("test","valid","train")):
    for sysn,st,lab,c in (("original","--","referência","#888"),("nosso","-","proposto","#0072B2")):
        if (sysn,sp) not in dec: continue
        y=dec[(sysn,sp)]; ax.plot(niv,y,st,color=c,lw=1.8,marker="o",ms=4,label=lab)
        ax.annotate(f"{y[-1]:.1f}",(3,y[-1]),xytext=(5,0),textcoords="offset points",va="center",fontsize=8,color=c)
    ax.set_title(NOME[sp],loc="left",fontsize=9.5,fontweight="bold"); ax.grid(axis="y",color="#eee",lw=0.6); ax.set_ylim(0,70); ax.tick_params(axis="x",labelrotation=25,labelsize=8)
axs[0].set_ylabel("GS-HOTA (%)"); axs[0].legend(frameon=False,fontsize=8,loc="lower left")
fig.tight_layout(w_pad=1.0); fig.savefig(f"{OUT}/fig_decomposicao_gshota.pdf"); fig.savefig(f"{OUT}/fig_decomposicao_gshota.png",dpi=200); plt.close(fig)

# 3) frames baseline vs nosso (SNGS-148, mesmo instante)
FF="/home/joao/soccernet/ffbin/ffmpeg"; T="00:00:12"; A="/home/joao/soccernet/baseline_out/apresentacao"
for tag,f in (("base",f"{A}/SNGS-148_baseline_original_100pct.mp4"),("nosso",f"{A}/SNGS-148_nosso.mp4")):
    subprocess.run([FF,"-y","-loglevel","error","-ss",T,"-i",f,"-frames:v","1",f"/dev/shm/f148_{tag}.png"],check=True)
ims=[Image.open(f"/dev/shm/f148_{t}.png").convert("RGB").crop((380,520,1700,1080)) for t in ("base","nosso")]
fig,axs=plt.subplots(1,2,figsize=(7.2,1.9))
for ax,im,t in zip(axs,ims,("Solução de referência (lado trocado)","Sistema proposto")):
    ax.imshow(im); ax.set_title(t,fontsize=9,loc="left"); ax.axis("off")
fig.tight_layout(w_pad=0.6); fig.savefig(f"{OUT}/fig_frames_148.pdf",dpi=220); fig.savefig(f"{OUT}/fig_frames_148.png",dpi=220); plt.close(fig)
for f in ("fig_delta_por_video.pdf","fig_decomposicao_gshota.pdf","fig_frames_148.pdf"): subprocess.run(["cp",f"{OUT}/{f}",TOUT])
print("ok")
