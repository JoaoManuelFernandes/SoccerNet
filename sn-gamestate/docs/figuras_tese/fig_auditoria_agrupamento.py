"""Figura: agrupamento em equipas com auditoria (módulo team). Frequências reais:
treino 57 vídeos (docs/render_logs), teste 39 vídeos (docs/team_side_report/REPORT.md)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

AZUL = "#2a78d6"
INK, INK2, GRID, BG, PALE = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb", "#e9f1fb"

fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 120); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, title, body="", ec=INK2, fc="white", lw=1.2, tc=INK, bc=INK, fs=8.2, ts=9.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw, zorder=2))
    if body:
        ax.text(x+w/2, y+h-1.5, title, ha="center", va="top", fontsize=ts, fontweight="bold", color=tc, zorder=3)
        ax.text(x+w/2, y+h-5.4, body, ha="center", va="top", fontsize=fs, color=bc, zorder=3, linespacing=1.35)
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=tc, zorder=3)
def diamond(cx, cy, w, h, text, fc=PALE, ec=AZUL):
    ax.add_patch(plt.Polygon([(cx-w/2, cy), (cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2)], closed=True, fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.0, color=INK, zorder=3, linespacing=1.35)
def arr(p0, p1, c=INK2, lw=1.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=c, lw=lw, zorder=4))
def lab(x, y, t, c=INK2, rot=0, ha="left"):
    ax.text(x, y, t, fontsize=8.0, color=c, ha=ha, va="center", fontweight="bold", rotation=rot, zorder=5)

ax.text(1.5, 99, "Agrupamento em duas equipas com auditoria (módulo team)", fontsize=11, fontweight="bold", color=INK, va="top")
ax.text(1.5, 95.6, "por vídeo, sobre os tracklets com função 'jogador'", fontsize=8.6, color=INK2, va="top")

CX = 38; L, W = 8, 60
box(L, 80, W, 11, "1. Uma linha por tracklet",
    "média das deteções do tracklet: embedding ReID (256 d), descritor de\ncor (27 d, calculado no módulo reid) e posição x no campo (|x| ≤ 52.5 m)")
arr((CX, 80), (CX, 75.2))
box(L, 64, W, 11, "2. k-means (k = 2) sobre o ReID (256 d)",
    "sinal principal: aparência aprendida; sobre os dois grupos\nmede-se a silhueta s e a distância L2 entre os centróides")
arr((CX, 64), (CX, 59.2))
diamond(CX, 49, 58, 20, "3. Auditoria: o ReID separou mal?\n\n"
        "gatilho A: s < 0.28\n"
        "gatilho B (herdado): ≥ 85% dos tracklets na mesma\nmetade do campo  e  L2 < 2.04\n"
        "basta um dos dois")
# não → manter ReID (direita)
arr((CX+29, 49), (78, 49)); lab(CX+30.5, 50.8, "não")
box(78, 44, 22, 10, "manter ReID", "grupos do passo 2", ec=GRID, tc=INK2, bc=INK2, ts=9.0, fs=7.8)
# sim → cor
arr((CX, 39), (CX, 33.2), c=AZUL); lab(CX+1.5, 36, "sim", c=AZUL)
box(L, 22, W, 11, "4. k-means (k = 2) sobre a cor (27 d, normalizada por canal)",
    "só a cauda de cor: misturada com os 256 d do ReID, a cor fica abafada;\nguarda: o grupo menor tem de ter ≥ 20% dos tracklets (só gatilho A)",
    ec=AZUL, fc=PALE, lw=1.6)
# guarda passou → usar cor (baixo)
arr((CX, 22), (CX, 15.2), c=AZUL); lab(CX+1.5, 18.6, "guarda passou", c=AZUL)
box(CX-11, 6, 22, 9, "usar cor", "grupos do passo 4", ec=AZUL, fc=PALE, ts=9.0, fs=7.8)
# guarda falhou → manter ReID (sobe pela direita)
ax.plot([L+W, 89], [27.5, 27.5], color=INK2, lw=1.1, zorder=4)
arr((89, 27.5), (89, 44)); lab(90.5, 35, "guarda falhou", rot=90)
ax.text(1.5, 1.8, "saída, em qualquer dos ramos: team_cluster ∈ {0, 1} por tracklet; o lado (left/right) é decidido a seguir, no módulo team_side",
        fontsize=8.2, color=INK2, va="bottom", style="italic")

# painel de frequências
XP, YP, WP, HP = 74, 60, 44, 32
ax.add_patch(FancyBboxPatch((XP, YP), WP, HP, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=GRID, lw=1.0, zorder=2))
ax.text(XP+WP/2, YP+HP-1.5, "Frequências medidas", ha="center", va="top", fontsize=9.4, fontweight="bold", color=INK, zorder=3)
txt = ("treino, 57 vídeos (registos do render)\n"
       "  manter ReID 49 · usar cor 8\n"
       "  dos 8: gatilho B 7, só gatilho A 1 (069)\n"
       "  guarda falhou 1 (061, grupo menor 14%)\n"
       "  s: 0.234 a 0.456 (mediana 0.391)\n"
       "  L2: 1.467 a 2.681 (mediana 2.261)\n\n"
       "teste, 39 vídeos com estado guardado\n"
       "  manter ReID 35 · usar cor 4\n"
       "  (136, 141, 195, 197; L2 1.799 a 2.035)\n\n"
       "o limiar 2.04 fica 0.005 acima do maior\n"
       "L2 desses 4 vídeos: foi ajustado a eles")
ax.text(XP+1.8, YP+HP-5.0, txt, ha="left", va="top", fontsize=7.6, color=INK, family="monospace", linespacing=1.38, zorder=3)

fig.savefig("fig_auditoria_agrupamento.png", bbox_inches="tight"); fig.savefig("fig_auditoria_agrupamento.pdf", bbox_inches="tight")
print("ok")
