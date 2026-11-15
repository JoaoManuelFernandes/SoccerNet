"""Figura: de onde vêm os 256 erros da baseline (39 vídeos de teste)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

AZUL = "#2a78d6"; INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
cats = ["Inversão do lado\n(5 vídeos a 0%)", "Agrupamento misturado\n(1 vídeo a 53.8%)", "Residual do tracker\n(4 vídeos, 1 tracklet cada)"]
vals = [228, 24, 4]; tot = sum(vals)

fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=200)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
y = [2, 1, 0]
for yi, v in zip(y, vals):
    ax.add_patch(FancyBboxPatch((0, yi-0.32), v, 0.64, boxstyle="round,pad=0,rounding_size=2",
                                fc=AZUL, ec="none", mutation_aspect=0.02))
    ax.text(v + 3, yi, f"{v} tracklets   ({v/tot*100:.1f}%)", va="center", ha="left",
            fontsize=10, color=INK, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=9.5, color=INK)
ax.set_xlim(0, 300); ax.set_ylim(-0.6, 2.6)
ax.set_xlabel("tracklets mal atribuídos na solução de referência (total 256)", fontsize=9, color=INK2)
ax.tick_params(axis="x", colors=INK2, labelsize=8.5); ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
fig.tight_layout()
fig.savefig("sn-gamestate/docs/figuras_tese/fig_erros_baseline.png", bbox_inches="tight")
fig.savefig("sn-gamestate/docs/figuras_tese/fig_erros_baseline.pdf", bbox_inches="tight")
print("ok")
