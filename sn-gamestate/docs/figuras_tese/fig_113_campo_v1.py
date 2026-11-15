"""Figura: SNGS-113 sobre o campo, anotação de referência. Um marcador por trajetória anotada
(posição mediana no clip de 30 s)."""
import json, statistics as st, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

AZUL, LARANJA = "#2a78d6", "#eb6834"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"

d = json.load(open("/home/joao/soccernet/data/SoccerNetGS/train/SNGS-113/Labels-GameState.json"))
per = collections.defaultdict(list); allx = {"left": [], "right": []}
for a in d["annotations"]:
    at = a.get("attributes") or {}; p = a.get("bbox_pitch") or {}
    x, y = p.get("x_bottom_middle"), p.get("y_bottom_middle")
    if x is None or y is None or abs(x) > 60: continue
    r, t = at.get("role"), at.get("team")
    if t not in allx or r not in ("player", "goalkeeper"): continue
    per[(a["track_id"], r, t)].append((x, y))
    if r == "player": allx[t].append(x)
med = {t: st.median(v) for t, v in allx.items()}
gk = {t: st.median(x for x, _ in v) for (tid, r, t), v in per.items() if r == "goalkeeper"}

fig, ax = plt.subplots(figsize=(11, 8.6), dpi=200)
ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
L, W = 52.5, 34
ax.add_patch(Rectangle((-L, -W), 2*L, 2*W, fill=False, lw=1.2, ec=INK2))
ax.plot([0, 0], [-W, W], lw=1, color=INK2); ax.add_patch(Circle((0, 0), 9.15, fill=False, lw=1, ec=INK2))
for s in (-1, 1):
    ax.add_patch(Rectangle((s*L - (16.5 if s > 0 else 0), -20.16), 16.5, 40.32, fill=False, lw=1, ec=INK2))
    ax.add_patch(Rectangle((s*L - (5.5 if s > 0 else 0), -9.16), 5.5, 18.32, fill=False, lw=1, ec=INK2))
    ax.plot([s*L, s*L], [-3.66, 3.66], lw=3, color=INK2, solid_capstyle="butt")
ax.text(-L+1, W-2, "baliza esquerda", fontsize=7.5, color=INK2, ha="left", va="top")
ax.text(L-1, -W+1.5, "as posições são as da anotação de referência (projeção anotada no campo);\nduas trajetórias da equipa esq. ficam sobre a linha lateral superior, com y de +34.1 m e +36.2 m",
        fontsize=6.8, color=INK2, ha="right", va="bottom", style="italic")
ax.text(L-1, W-2, "baliza direita", fontsize=7.5, color=INK2, ha="right", va="top")

# um marcador por trajetória anotada
for (tid, r, t), v in per.items():
    mx, my = st.median(x for x, _ in v), st.median(y for _, y in v)
    c = AZUL if t == "left" else LARANJA
    if r == "player":
        ax.scatter([mx], [my], s=95, color=c, edgecolor="white", lw=1.0, zorder=5)
    else:
        ax.scatter([mx], [my], s=230, marker="D", color=c, edgecolor="white", lw=1.5, zorder=6)
        ax.text(mx, my-4.2, f"GR {'esq.' if t=='left' else 'dir.'}\nx = {mx:+.1f} m", ha="center", va="top", fontsize=8.6,
                color=c, fontweight="bold", zorder=7)

# medianas de equipa (linhas) e as duas distâncias
for t, c in (("left", AZUL), ("right", LARANJA)):
    ax.plot([med[t], med[t]], [-W, W], color=c, lw=1.8, ls=(0, (5, 3)), zorder=4)

yA = -W-4.5
ax.annotate("", xy=(med["right"], yA), xytext=(med["left"], yA), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
ax.text(med["left"]-1.5, yA, f"mediana equipa esq.\n{med['left']:+.1f} m", ha="right", va="center", fontsize=8.6, color=AZUL, fontweight="bold")
ax.text(med["right"]+1.5, yA, f"mediana equipa dir.\n{med['right']:+.1f} m", ha="left", va="center", fontsize=8.6, color=LARANJA, fontweight="bold")
ax.text((med["left"]+med["right"])/2, yA-1.6, f"{abs(med['right']-med['left']):.1f} m", ha="center", va="top", fontsize=9, color=INK, fontweight="bold")
yB = -W-12.5
ax.annotate("", xy=(gk["right"], yB), xytext=(gk["left"], yB), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
ax.text((gk["left"]+gk["right"])/2, yB-1.3, f"guarda-redes: {abs(gk['right']-gk['left']):.1f} m", ha="center", va="top", fontsize=9, color=INK, fontweight="bold")

# o que isto significa para a cascata
txt = ("O que a cascata compara (critérios 1 a 3): a posição dos jogadores de cada grupo.\n"
       "Aqui as duas equipas estão na mesma metade e as medianas distam 9.7 m: o sinal é fraco.\n"
       "O que a cascata só consulta no critério 4: o guarda-redes. Aqui os dois estão a 73.5 m\n"
       "um do outro, o sinal é forte, mas o critério 2 decidiu primeiro (ver figura da cascata).")
ax.text(-L, -W-19.5, txt, ha="left", va="top", fontsize=8.6, color=INK, linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=GRID))

ax.set_xlim(-L-2, L+2); ax.set_ylim(-W-33, W+8); ax.set_aspect("equal"); ax.axis("off")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=AZUL, ms=8, label="jogador, equipa esq. (baliza à esquerda)"),
                   Line2D([], [], marker="o", ls="", color=LARANJA, ms=8, label="jogador, equipa dir. (baliza à direita)"),
                   Line2D([], [], marker="D", ls="", color=INK2, ms=8, label="guarda-redes")],
          loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8, frameon=False, ncol=3, handletextpad=0.4, columnspacing=1.2)
ax.set_title("SNGS-113 (treino), anotação de referência: um marcador por trajetória, posição mediana nos 30 s do clip",
             fontsize=9.6, color=INK, loc="left", pad=22)
fig.savefig("fig_113_campo.png", bbox_inches="tight"); fig.savefig("fig_113_campo.pdf", bbox_inches="tight")
print("ok", med, gk)
