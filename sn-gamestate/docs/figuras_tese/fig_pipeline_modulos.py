"""Figura: Pipeline GSR (módulos). Nove módulos na ordem real de configs/soccernet.yaml,
duas fases (por imagem/deteção; por vídeo), os quatro módulos tocados em destaque."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe

AZUL, AZUL_ESC = "#2a78d6", "#1b56a3"
INK, INK2, GRID, BG, PALE, SOMBRA = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb", "#e9f1fb", "#d9dcdf"

mods = [
 ("bbox_detector", "Deteção de pessoas", None),
 ("reid", "Reidentificação", "estendido"),
 ("track", "Seguimento", None),
 ("pitch", "Pontos do campo", None),
 ("calibration", "Calibração", None),
 ("jersey_number_detect", "Número da camisola", "substituído"),
 ("tracklet_agg", "Agregação", None),
 ("team", "Equipa", "reescrito"),
 ("team_side", "Lado do campo", "reescrito"),
]
fig, ax = plt.subplots(figsize=(13, 6.0), dpi=220)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 132); ax.set_ylim(0, 60); ax.axis("off")

W, H, GAP = 23.4, 15.0, 2.7
rows = {0: 32.0, 1: 7.0}
x0 = {0: 1.5, 1: 1.5 + (W + GAP) / 2}
pos = []
for i, (key, nome, estado) in enumerate(mods):
    r = 0 if i < 5 else 1; c = i if r == 0 else i - 5
    x, y = x0[r] + c * (W + GAP), rows[r]; pos.append((x, y))
    changed = estado is not None
    ax.add_patch(FancyBboxPatch((x + 0.5, y - 0.6), W, H, boxstyle="round,pad=0,rounding_size=1.6", fc=SOMBRA, ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=PALE if changed else "white", ec=AZUL if changed else "#c9ccd1", lw=1.6 if changed else 1.0, zorder=2))
    if changed:
        ax.add_patch(FancyBboxPatch((x, y + H - 1.3), W, 1.3, boxstyle="round,pad=0,rounding_size=0.6", fc=AZUL, ec="none", zorder=3))
    ax.add_patch(Circle((x + 2.7, y + H - 3.4), 1.5, fc=AZUL if changed else "#eceef1", ec="none", zorder=3))
    ax.text(x + 2.7, y + H - 3.4, str(i + 1), ha="center", va="center", fontsize=7.4, fontweight="bold", color="white" if changed else INK2, zorder=4)
    ax.text(x + W / 2, y + H - 7.0, nome, ha="center", va="center", fontsize=9.0, fontweight="bold", color=INK if changed else "#2b2b2b", zorder=4)
    ax.text(x + W / 2, y + 4.2, key, ha="center", va="center", fontsize=7.4, family="monospace", color=AZUL_ESC if changed else INK2, zorder=4)
    if changed:
        ax.text(x + W / 2, y + 1.6, estado, ha="center", va="center", fontsize=7.4, color=AZUL_ESC, fontweight="bold", zorder=4)

def seta(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color="#7a7d82", lw=1.2, zorder=5, shrinkA=0, shrinkB=0))
for i in range(len(mods) - 1):
    (xa, ya), (xb, yb) = pos[i], pos[i + 1]
    if i == 4:
        cx = xa + W / 2; cy = ya - 3.4; lx_ = xb - 3.2; my_ = yb + H / 2
        ax.plot([cx, cx], [ya, cy], color="#7a7d82", lw=1.2, zorder=5)
        ax.plot([cx, lx_], [cy, cy], color="#7a7d82", lw=1.2, zorder=5)
        ax.plot([lx_, lx_], [cy, my_], color="#7a7d82", lw=1.2, zorder=5)
        seta((lx_, my_), (xb, my_))
    else:
        seta((xa + W, ya + H / 2), (xb, yb + H / 2))

# fases
ax.text(1.5, rows[0] + H + 2.2, "POR IMAGEM E POR DETEÇÃO", fontsize=7.6, color=INK2, va="bottom", fontweight="bold")
ax.plot([1.5, x0[0] + 4 * (W + GAP) + W], [rows[0] + H + 1.8, rows[0] + H + 1.8], color=GRID, lw=1.0)
ax.text(x0[1], rows[1] + H + 2.2, "POR VÍDEO, SOBRE TRACKLETS COMPLETOS", fontsize=7.6, color=INK2, va="bottom", fontweight="bold")
ax.plot([x0[1], x0[1] + 3 * (W + GAP) + W], [rows[1] + H + 1.8, rows[1] + H + 1.8], color=GRID, lw=1.0)

ax.text(1.5, 58.5, "Pipeline GSR (módulos)", fontsize=13, fontweight="bold", color=INK, va="top")
# legenda
lx, ly = 98.5, 55.5
ax.add_patch(FancyBboxPatch((lx, ly - 1.2), 3.2, 2.4, boxstyle="round,pad=0,rounding_size=0.5", fc=PALE, ec=AZUL, lw=1.4))
ax.text(lx + 4.2, ly, "módulo tocado neste trabalho", fontsize=8, color=INK2, va="center")
ax.add_patch(FancyBboxPatch((lx, ly - 4.8), 3.2, 2.4, boxstyle="round,pad=0,rounding_size=0.5", fc="white", ec="#c9ccd1", lw=1.0))
ax.text(lx + 4.2, ly - 3.6, "código e pesos originais", fontsize=8, color=INK2, va="center")

fig.savefig("fig_pipeline_modulos.png", bbox_inches="tight"); fig.savefig("fig_pipeline_modulos.pdf", bbox_inches="tight")
print("ok")
