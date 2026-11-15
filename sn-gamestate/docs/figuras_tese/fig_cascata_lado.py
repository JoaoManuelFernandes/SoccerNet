"""Figura: cascata de critérios do módulo team_side, com o rasto real do SNGS-113
(fonte: docs/render_logs/SNGS-113.log, linhas 3206-3234)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

AZUL, LARANJA = "#2a78d6", "#eb6834"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"
PALE = "#e9f1fb"

fig, ax = plt.subplots(figsize=(13, 8.4), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 118); ax.set_ylim(0, 87); ax.axis("off")

def box(x, y, w, h, title, body, ec=INK2, fc="white", lw=1.2, title_c=INK, body_c=INK2, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x+1.6, y+h-1.6, title, ha="left", va="top", fontsize=9.6, fontweight="bold", color=title_c, zorder=3)
    ax.text(x+1.6, y+h-6.0, body, ha="left", va="top", fontsize=fs, color=body_c, zorder=3, linespacing=1.35)

def arrow(x0, y0, x1, y1, c=INK2, lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=c, lw=lw, linestyle=ls, zorder=4))

# ---------------- coluna esquerda: a cascata ----------------
X, W = 3, 54
ax.text(X, 85.5, "Cascata de decisão do lado (módulo team_side)", fontsize=11, fontweight="bold", color=INK, va="top")
ax.text(X, 81.8, "Entrada: as equipas A e B, os dois grupos de tracklets que o módulo team separou pela aparência (k-means).", fontsize=8.2, color=INK2, va="top")
ax.text(X, 79.4, "Sabemos que vestem igual, mas não que baliza defendem. Descobrir qual é left e qual é right é o que esta cascata faz.", fontsize=8.2, color=INK2, va="top")

steps = [
  (62.5, "1. Votação por tracklet",
   "cada tracklet vota right se a sua média de x > 0, senão left;\n"
   "um grupo fica decidido se ≥ 60% dos seus tracklets concordam;\n"
   "falha se nenhum grupo decide ou se os dois saem para o mesmo lado"),
  (46.0, "2. Votação relativa",
   "compara a fração de tracklets à direita de cada grupo;\n"
   "se a diferença ≥ 10 pp, o grupo com mais tracklets à direita é right;\n"
   "falha se a diferença for menor que 10 pp"),
  (29.5, "3. Média de x por grupo",
   "grupo com maior média de x (deteções com |x| ≤ 60 m) é right;\n"
   "se as duas médias distam menos de 0.3 m, o sinal é ruído e\n"
   "passa ao critério 4 (só se houver guarda-redes detetado)"),
  (13.0, "4. Desempate pelo guarda-redes",
   "lado do guarda-redes = lado da baliza que ele defende;\n"
   "o grupo cujos tracklets avançam para essa baliza é o atacante,\n"
   "logo o outro grupo é a equipa desse guarda-redes"),
]
H = 13.0
for i, (y, t, b) in enumerate(steps):
    taken = i < 2
    box(X, y, W, H, t, b, ec=AZUL if taken else GRID, fc=PALE if taken else "white",
        lw=1.6 if taken else 1.0, title_c=INK if taken else INK2, body_c=INK if taken else INK2)
    if i < 3:
        arrow(X+W/2, y, X+W/2, y-3.5+0.1, c=AZUL if i == 0 else GRID)
        ax.text(X+W/2+1.2, y-1.8, "falhou", fontsize=7.8, color=AZUL if i == 0 else INK2, va="center")

# saída
ax.add_patch(FancyBboxPatch((X, 3.0), W, 6.0, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=GRID, lw=1.0, zorder=2))
ax.text(X+W/2, 6.0, "saída: A → left ou right, B → o oposto; guarda-redes rotulados pelo próprio x", ha="center", va="center", fontsize=8.2, color=INK2, zorder=3)
arrow(X+W/2, 13.0, X+W/2, 9.3, c=GRID)

# ---------------- coluna direita: rasto SNGS-113 ----------------
XR, WR = 61, 52
ax.text(XR, 85.5, "Rasto real no SNGS-113 (treino, registo do render de 2026-09-09)", fontsize=11, fontweight="bold", color=INK, va="top")

trace = [
  (62.5, True,  "Equipa A: 33 tracklets à direita, 8 à esquerda → 80% ≥ 60% → right\n"
                "Equipa B: 23 à direita, 1 à esquerda → 96% ≥ 60% → right\n"
                "as duas saem right → conflito, falhou"),
  (46.0, True,  "A = 80% à direita, B = 96% à direita, diferença 15 pp ≥ 10 pp\n"
                "decisão: B → right, A → left\n"
                "(assume que o grupo mais à direita defende a baliza direita)"),
  (29.5, False, "não atingido\n(medianas das equipas na anotação: +7.9 m e +17.6 m,\nas duas na metade direita, separadas por 9.7 m)"),
  (13.0, False, "não atingido\n(havia guarda-redes detetado: 457 deteções, lado left)"),
]
for (y, hit, b) in trace:
    ax.add_patch(FancyBboxPatch((XR, y), WR, H, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=PALE if hit else "white", ec=AZUL if hit else GRID, lw=1.6 if hit else 1.0, zorder=2))
    ax.text(XR+1.6, y+H-2.2, b, ha="left", va="top", fontsize=8.2, color=INK if hit else INK2, zorder=3, linespacing=1.4)

# marcador da decisão errada
ax.add_patch(FancyBboxPatch((XR, 3.0), WR, 6.0, boxstyle="round,pad=0,rounding_size=1.2", fc="#fdeee8", ec=LARANJA, lw=1.4, zorder=2))
ax.text(XR+WR/2-2, 6.0, "resultado: lados trocados\n(veredicto visual no vídeo renderizado)", ha="center", va="center", fontsize=8.4, color=INK, fontweight="bold", zorder=3)
# seta pela margem direita, do critério 2 até ao resultado
xe = XR+WR+2.5
ax.plot([XR+WR, xe], [52.5, 52.5], color=LARANJA, lw=1.5, zorder=4)
arrow(xe, 52.5, xe, 6.0, c=LARANJA, lw=1.5)
ax.plot([xe, XR+WR], [6.0, 6.0], color=LARANJA, lw=1.5, zorder=4)
ax.text(xe+1.0, 30.0, "decidiu aqui", fontsize=8.2, color=LARANJA, va="center", ha="left", fontweight="bold", rotation=90)

fig.savefig("fig_cascata_lado.png", bbox_inches="tight"); fig.savefig("fig_cascata_lado.pdf", bbox_inches="tight")
print("ok")
