"""Figura do Capítulo 4: exatidão da equipa por vídeo, baseline -> sistema proposto, só nos vídeos que mudam.
Lê nosso_out/TABELA_VIDEO_A_VIDEO{SUF}.csv. Gera fig_equipa_por_video.pdf (+PNG) e copia o PDF para TESE_OUT."""
import csv, os, subprocess
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
SUF=os.environ.get("SUF","_conf"); TOUT=os.environ.get("TESE_OUT","/home/joao/soccernet/tese")
NOSSO="/home/joao/soccernet/nosso_out"; OUT=os.environ.get("FIG_OUT",os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#777","axes.linewidth":0.6,"xtick.color":"#444","ytick.color":"#444","axes.labelcolor":"#222"})
COR={"test":"#0072B2","valid":"#E69F00","train":"#009E73"}; NOME={"test":"teste","valid":"validação","train":"treino"}
rows=list(csv.DictReader(open(f"{NOSSO}/TABELA_VIDEO_A_VIDEO{SUF}.csv")))

# Empate = diferença de até 1 ponto, como na contagem do texto (44 melhoram, 104 empatam, 11 pioram)
v=[(float(r["equipa_original"]),float(r["equipa_nosso"]),r["split"],r["video"][5:]) for r in rows]
muda=sorted((t for t in v if abs(t[1]-t[0])>1),key=lambda t:t[1]-t[0])
n_eq=len(v)-len(muda)

fig,ax=plt.subplots(figsize=(7.2,3.3))
for x,(o,n,s,_) in enumerate(muda):
    ax.plot([x,x],[o,n],color=COR[s],lw=1.6,solid_capstyle="butt",zorder=1)
    ax.scatter(x,o,s=16,facecolor="white",edgecolor=COR[s],linewidth=1.1,zorder=2)
    ax.scatter(x,n,s=16,color=COR[s],zorder=3)
n_dn=sum(1 for o,n,_,_ in muda if n<o)
ax.axvline(n_dn-0.5,color="#bbb",lw=0.6,ls=(0,(3,3)))
ax.text(n_dn/2-0.5,104,f"pioram ({n_dn})",ha="center",va="bottom",fontsize=8,color="#333")
ax.text((n_dn+len(muda))/2-0.5,104,f"melhoram ({len(muda)-n_dn})",ha="center",va="bottom",fontsize=8,color="#333")
ax.set_xticks(range(len(muda))); ax.set_xticklabels([t[3] for t in muda],rotation=90,fontsize=6)
ax.set_xlim(-0.8,len(muda)-0.2); ax.set_ylim(-4,112); ax.set_yticks([0,25,50,75,100])
ax.set_ylabel("Exatidão da equipa (%)"); ax.set_xlabel("vídeo (SNGS-)",fontsize=8)
ax.grid(axis="y",color="#e6e6e6",lw=0.5); ax.set_axisbelow(True)
leg=[Line2D([],[],color=COR[s],lw=1.6,label=NOME[s]) for s in ("test","valid","train")]
leg+=[Line2D([],[],ls="none",marker="o",ms=4.5,mfc="white",mec="#555",label="baseline"),
      Line2D([],[],ls="none",marker="o",ms=4.5,color="#555",label="sistema proposto")]
ax.legend(handles=leg,loc="lower left",bbox_to_anchor=(0.0,1.01),frameon=False,fontsize=7.5,ncol=5,handlelength=1.6,columnspacing=1.4)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_equipa_por_video.pdf"); fig.savefig(f"{OUT}/fig_equipa_por_video.png",dpi=200); plt.close(fig)
print(f"mudam {len(muda)} (pioram {n_dn}), empatam {n_eq}")
subprocess.run(["cp",f"{OUT}/fig_equipa_por_video.pdf",TOUT])
