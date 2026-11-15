"""Figura: três casos reais da cascata do lado, decididos em pontos diferentes.
SNGS-060 (treino, registo do render 2026-09-09), SNGS-195 (teste, reexecução dos módulos team e
team_side sobre o estado guardado, 2026-09-13), SNGS-113 (treino, registo do render 2026-09-09)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#2e8b57"
INK, INK2, GRID, BG, PALE = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb", "#e9f1fb"

casos = [
 ("SNGS-060 (treino)", "jogo aberto, equipas nas duas metades", 3, True,
  ["A: 15 à direita, 19 à esquerda → 44%, entre 40% e 60% → indecisa\nB: 22 à direita, 20 à esquerda → 52% → indecisa\nninguém decide → falhou",
   "A = 44% à direita, B = 52% → diferença 8 pp < 10 pp\n→ sem sinal suficiente, falhou",
   "média de x: A = −2.0 m, B = +2.4 m → distam 4.4 m ≥ 0.3 m\n→ B é right, A é left",
   "não atingido"],
  "resultado: correto (veredicto visual)"),
 ("SNGS-195 (teste)", "jogo todo na metade esquerda", 4, True,
  ["A: 4 à direita, 16 à esquerda → 80% esquerda → left\nB: 5 à direita, 30 à esquerda → 86% esquerda → left\nas duas saem left → conflito, falhou",
   "A = 20% à direita, B = 14% → diferença 6 pp < 10 pp\n→ falhou",
   "média de x: A = −23.03 m, B = −23.04 m → distam 0.01 m,\nmenos de 0.3 m → sem sinal, passa ao guarda-redes",
   "guarda-redes detetado à esquerda (x ≈ −46 m)\ndeslocamento médio para essa baliza: A = 1.4 m, B = 5.8 m\nB avança mais → B ataca a baliza esquerda → B é right, A é left"],
  "resultado: correto (31/31 tracklets)"),
 ("SNGS-113 (treino)", "jogo todo na metade direita", 2, False,
  ["A: 33 à direita, 8 à esquerda → 80% → right\nB: 23 à direita, 1 à esquerda → 96% → right\nas duas saem right → conflito, falhou",
   "A = 80% à direita, B = 96% → diferença 15 pp ≥ 10 pp\n→ B é right, A é left",
   "não atingido",
   "não atingido (havia guarda-redes detetado à esquerda)"],
  "resultado: invertido (veredicto visual)"),
]
passos = ["1. Votação por tracklet", "2. Votação relativa", "3. Média de x", "4. Desempate pelo guarda-redes"]

fig, ax = plt.subplots(figsize=(14, 9.6), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 140); ax.set_ylim(4, 96); ax.axis("off")
ax.text(1.5, 95, "A cascata do lado em três vídeos reais: decide no critério 3, no 4 e, mal, no 2", fontsize=11.5, fontweight="bold", color=INK, va="top")
ax.text(1.5, 91.6, "A e B são as duas equipas que saem do agrupamento; x é a posição no campo (−52.5 m baliza esquerda, +52.5 m baliza direita)", fontsize=8.4, color=INK2, va="top")

CW, X0, GAPX = 45, 1.0, 2.0
H = 13.5; Y0 = 69.5; GAPY = 3.4
for k, (titulo, contexto, decide, correto, textos, res) in enumerate(casos):
    x = X0 + k * (CW + GAPX)
    ax.text(x, 88.5, titulo, fontsize=10, fontweight="bold", color=INK, va="top")
    ax.text(x, 85.6, contexto, fontsize=8.2, color=INK2, va="top", style="italic")
    for i, (p, t) in enumerate(zip(passos, textos)):
        y = Y0 - i * (H + GAPY)
        atingido = i < decide
        ec = (AZUL if atingido else GRID); fc = (PALE if atingido else "white")
        if i == decide - 1: ec = (VERDE if correto else LARANJA); lw = 2.0
        else: lw = 1.4 if atingido else 1.0
        ax.add_patch(FancyBboxPatch((x, y), CW, H, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + 1.4, y + H - 1.4, p, fontsize=8.6, fontweight="bold", color=INK if atingido else INK2, va="top", zorder=3)
        ax.text(x + 1.4, y + H - 4.6, t, fontsize=7.2, color=INK if atingido else INK2, va="top", zorder=3, linespacing=1.35)
        if i == decide - 1:
            ax.text(x + CW - 1.2, y + H - 1.4, "decidiu aqui", fontsize=7.6, fontweight="bold", color=VERDE if correto else LARANJA, ha="right", va="top", zorder=3)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + CW / 2, y), (x + CW / 2, y - GAPY + 0.3), arrowstyle="-|>", mutation_scale=9,
                                         color=AZUL if i < decide - 1 else GRID, lw=1.1, zorder=4))
    yr = Y0 - 3 * (H + GAPY) - 6.2
    c = VERDE if correto else LARANJA
    ax.add_patch(FancyBboxPatch((x, yr), CW, 4.4, boxstyle="round,pad=0,rounding_size=1.0", fc="#eaf5ee" if correto else "#fdeee8", ec=c, lw=1.4, zorder=2))
    ax.text(x + CW / 2, yr + 2.2, res, fontsize=8.4, fontweight="bold", color=INK, ha="center", va="center", zorder=3)

fig.savefig("fig_cascata_exemplos.png", bbox_inches="tight"); fig.savefig("fig_cascata_exemplos.pdf", bbox_inches="tight")
print("ok")
