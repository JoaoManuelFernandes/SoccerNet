"""Figura: construção do descritor de cor (27 d) no módulo reid.
Recorte real (SNGS-116, imagem 1, jogador da equipa left, n.º 93). A máscara do tronco é
um ESQUEMA (o mapa de atenção do BPBreID não é guardado no estado); os histogramas são
calculados aqui sobre a banda central (25%-75% da altura), que é o modo de recurso do
código quando a máscara tem menos de 16 píxeis, e servem só para ilustrar o formato."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
import numpy as np, cv2

AZUL = "#2a78d6"
INK, INK2, GRID, BG, PALE = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb", "#e9f1fb"

img = cv2.imread("/home/joao/soccernet/data/SoccerNetGS/test/SNGS-116/img1/000001.jpg")
x, y, w, h = 478, 565, 51, 140
crop_bgr = img[y:y+h, x:x+w]; crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV); crop_lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2Lab)
mask = np.zeros((h, w), np.uint8); mask[int(0.25*h):int(0.75*h), :] = 1   # banda de recurso
hists = []
for ch, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
    hh = cv2.calcHist([crop_hsv], [ch], mask, [8], [lo, hi]).flatten(); hists.append(hh / hh.sum())
lab_mean = crop_lab[mask > 0].mean(0) / 255.0

fig = plt.figure(figsize=(12.5, 5.6), dpi=200); fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.0, 1.0, 2.2, 1.6], height_ratios=[1, 1], wspace=0.35, hspace=0.55,
                      left=0.03, right=0.98, top=0.93, bottom=0.10)

def panel(ax, title):
    ax.set_facecolor(BG); ax.set_title(title, fontsize=8.6, color=INK, pad=6, loc="left")
    for s in ax.spines.values(): s.set_visible(False)

# 1 recorte
ax1 = fig.add_subplot(gs[:, 0]); panel(ax1, "1. recorte da deteção"); ax1.imshow(crop_rgb); ax1.set_xticks([]); ax1.set_yticks([])
# 2 região do tronco (esquema; sem título nem subtítulo dentro da figura desde 12/10: a legenda da tese chega)
ax2 = fig.add_subplot(gs[:, 1]); panel(ax2, "2. região do tronco (esquema)"); ax2.imshow(crop_rgb, alpha=0.35)
ax2.add_patch(Ellipse((w/2, 0.40*h), 0.85*w, 0.42*h, fc=AZUL, ec="none", alpha=0.55))
ax2.text(w/2, 0.40*h, "tronco", ha="center", va="center", fontsize=7.4, color="white", fontweight="bold")
ax2.set_xlim(0, w); ax2.set_ylim(h, 0); ax2.set_xticks([]); ax2.set_yticks([])
# 3 píxeis selecionados
ax3 = fig.add_subplot(gs[:, 2]); panel(ax3, "3. píxeis selecionados"); sel = crop_rgb.copy(); sel[mask == 0] = (235, 235, 232)
ax3.imshow(sel); ax3.set_xticks([]); ax3.set_yticks([])
ax3.text(w/2, h*0.96, "atenção ≥ 0.25 do máximo;\nrecurso: banda 25%-75%", ha="center", va="bottom", fontsize=6.6, color=INK2)
# 4 histogramas
names = ["H (matiz)", "S (saturação)", "V (valor)"]
axh = fig.add_subplot(gs[0, 3]); panel(axh, "4. histogramas HSV, 8 classes cada, normalizados (24 d)")
xs = np.arange(24); cols = [AZUL]*8 + ["#6ea0e3"]*8 + ["#a9c6ef"]*8
axh.bar(xs, np.concatenate(hists), color=cols, width=0.8); axh.set_xticks([3.5, 11.5, 19.5]); axh.set_xticklabels(names, fontsize=7.2, color=INK2)
axh.tick_params(axis="y", labelsize=6.8, colors=INK2, length=0); axh.grid(axis="y", color=GRID, lw=0.5); axh.set_axisbelow(True)
for i in (8, 16): axh.axvline(i-0.5, color=GRID, lw=0.8)
# 5 Lab
axl = fig.add_subplot(gs[1, 3]); panel(axl, "5. média Lab dos mesmos píxeis, normalizada a [0, 1] (3 d)")
axl.barh(["L", "a", "b"][::-1], lab_mean[::-1], color=AZUL, height=0.55)
axl.set_xlim(0, 1); axl.tick_params(labelsize=7.2, colors=INK2, length=0); axl.grid(axis="x", color=GRID, lw=0.5); axl.set_axisbelow(True)
# 6 concatenação
ax6 = fig.add_subplot(gs[:, 4]); panel(ax6, "6. vetor final por deteção"); ax6.set_xlim(0, 10); ax6.set_ylim(0, 10); ax6.set_xticks([]); ax6.set_yticks([])
ax6.add_patch(FancyBboxPatch((1, 6.2), 8, 2.2, boxstyle="round,pad=0,rounding_size=0.3", fc="white", ec=INK2, lw=1.1))
ax6.text(5, 7.3, "embeddings\nReID, 256 d (inalterado)", ha="center", va="center", fontsize=7.4, color=INK)
ax6.add_patch(FancyBboxPatch((1, 2.2), 8, 2.8, boxstyle="round,pad=0,rounding_size=0.3", fc=PALE, ec=AZUL, lw=1.4))
ax6.text(5, 3.6, "embeddings_with_colors\n256 d ReID + 24 d HSV + 3 d Lab\n= 283 d (coluna nova)", ha="center", va="center", fontsize=7.4, color=INK)
ax6.add_patch(FancyArrowPatch((5, 6.2), (5, 5.1), arrowstyle="-|>", mutation_scale=10, color=INK2, lw=1.0))
ax6.text(5, 0.9, "o agrupamento em equipas usa só\nestes 27 d (24 HSV + 3 Lab)\nquando audita o ReID", ha="center", va="center", fontsize=7.0, color=INK2, style="italic")

fig.savefig("fig_descritor_cor.png", bbox_inches="tight"); fig.savefig("fig_descritor_cor.pdf", bbox_inches="tight")
print("ok", lab_mean.round(3))
