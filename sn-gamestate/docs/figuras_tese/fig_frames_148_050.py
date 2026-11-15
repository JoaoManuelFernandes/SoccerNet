"""Figuras do Capítulo 4 com frames dos vídeos renderizados (regerados a 2026-11-01 por baseline_out/tools/refaz_148.sh).
fig_frames_148: SNGS-148, frame 565 (ataque junto à baliza esquerda), baseline (lado trocado) e sistema proposto (certo),
  tal como sai do render, com o minimapa (predições à esquerda, anotação à direita). O frame foi escolhido por ter o
  minimapa destapado e a calibração boa nos dois sistemas (erro mediano 2.3 m face à anotação, 14 jogadores).
fig_arbitragem_050: SNGS-050, frame 450, sistema proposto (a baseline falha da mesma forma): equipa de amarelo dada como árbitros.
Gera PDF+PNG e copia os PDF para TESE_OUT."""
import os, subprocess
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
FF="/home/joao/soccernet/ffbin/ffmpeg"; V="/home/joao/soccernet/baseline_out/videos"
OUT=os.environ.get("FIG_OUT",os.path.dirname(os.path.abspath(__file__))); TOUT=os.environ.get("TESE_OUT","/home/joao/soccernet/tese")
TMP=os.environ.get("TMPDIR","/dev/shm")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})

def frame(video,n,box):
    p=f"{TMP}/frame_{os.path.basename(video)}_{n}.png"
    subprocess.run([FF,"-y","-loglevel","error","-i",video,"-vf",f"select=eq(n\\,{n})","-frames:v","1",p],check=True)
    return Image.open(p).convert("RGB").crop(box)

# SNGS-148: frame inteiro sem a bancada, com o minimapa
caixa=(0,330,1700,1080)
ims=[frame(f"{V}/SNGS-148_{s}.mp4",565,caixa) for s in ("baseline","nosso")]
fig,axs=plt.subplots(2,1,figsize=(7.2,6.6))
for ax,im,t in zip(axs,ims,("(a) Baseline","(b) Sistema proposto")):
    ax.imshow(im); ax.set_title(t,fontsize=9,loc="left"); ax.axis("off")
fig.tight_layout(h_pad=0.6); fig.savefig(f"{OUT}/fig_frames_148.pdf",dpi=250); fig.savefig(f"{OUT}/fig_frames_148.png",dpi=200); plt.close(fig)

# SNGS-050: só o sistema proposto
im=frame(f"{V}/SNGS-050_nosso.mp4",450,(180,345,1840,700))
fig,ax=plt.subplots(figsize=(7.2,1.56))
ax.imshow(im); ax.axis("off")
fig.tight_layout(pad=0.1); fig.savefig(f"{OUT}/fig_arbitragem_050.pdf",dpi=250); fig.savefig(f"{OUT}/fig_arbitragem_050.png",dpi=200); plt.close(fig)

for f in ("fig_frames_148.pdf","fig_arbitragem_050.pdf"): subprocess.run(["cp",f"{OUT}/{f}",TOUT])
print("ok")
