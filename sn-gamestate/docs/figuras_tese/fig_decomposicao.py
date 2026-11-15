"""Figura do Capítulo 4: GS-HOTA com os atributos ativados de forma progressiva (Tabela tab:decomposicao).
Os valores são os da tabela gerada a 2026-09-27 (tese/final_conf/Chapter4_final.tex); o ficheiro de origem
nosso_out/gshota/DECOMPOSICAO_ATRIBUTOS_conf.md já não existe, por isso ficam aqui escritos.
Gera fig_decomposicao_gshota.pdf (+PNG) e copia o PDF para TESE_OUT."""
import os, subprocess
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=os.environ.get("FIG_OUT",os.path.dirname(os.path.abspath(__file__))); TOUT=os.environ.get("TESE_OUT","/home/joao/soccernet/tese")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#777","axes.linewidth":0.6,"xtick.color":"#444","ytick.color":"#444","axes.labelcolor":"#222"})
DEC={("baseline","teste"):[61.8,59.4,53.8,31.3],("proposto","teste"):[61.8,59.4,57.4,47.9],
     ("baseline","validação"):[57.7,47.4,38.5,23.7],("proposto","validação"):[57.7,47.4,43.0,36.8],
     ("baseline","treino"):[58.3,58.2,56.1,31.5],("proposto","treino"):[58.3,58.2,55.8,45.4]}
niv=["posição","+ função","+ equipa","+ número"]
fig,axs=plt.subplots(1,3,figsize=(7.2,2.7),sharey=True)
for ax,sp in zip(axs,("teste","validação","treino")):
    for s,st,lab,c in (("baseline","--","baseline","#888"),("proposto","-","sistema proposto","#0072B2")):
        y=DEC[(s,sp)]; ax.plot(niv,y,st,color=c,lw=1.8,marker="o",ms=4,label=lab)
        ax.annotate(f"{y[-1]:.1f}",(3,y[-1]),xytext=(5,0),textcoords="offset points",va="center",fontsize=8,color=c)
    ax.set_title(sp,loc="left",fontsize=9.5,fontweight="bold"); ax.grid(axis="y",color="#eee",lw=0.6); ax.set_ylim(0,70); ax.tick_params(axis="x",labelrotation=25,labelsize=8)
axs[0].set_ylabel("GS-HOTA (%)"); axs[0].legend(frameon=False,fontsize=8,loc="lower left")
fig.tight_layout(w_pad=1.0); fig.savefig(f"{OUT}/fig_decomposicao_gshota.pdf"); fig.savefig(f"{OUT}/fig_decomposicao_gshota.png",dpi=200); plt.close(fig)
subprocess.run(["cp",f"{OUT}/fig_decomposicao_gshota.pdf",TOUT]); print("ok")
