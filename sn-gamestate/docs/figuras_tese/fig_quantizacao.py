"""Figura: quantização a 4 bits por blocos (bitsandbytes, FP4), num bloco REAL de 64 pesos do Qwen2.5-VL-3B
(primeira linha de model.layers.0.self_attn.q_proj.weight), quantizado com a própria biblioteca.
Painéis: (a) pesos originais e escala do bloco; (b) normalização e arredondamento para os 16 valores FP4;
(c) memória do modelo em 16 e 4 bits face aos 6 GB da GPU (valores medidos na RTX 3060 portátil).
Uso: venv_tl/bin/python fig_quantizacao.py -> fig_quantizacao.pdf/.png"""
import os, glob
import numpy as np, torch
import bitsandbytes.functional as F
from safetensors import safe_open
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, AZUL_ESC, LARANJA = "#2a78d6", "#1b56a3", "#e8692c"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/*/"))[0]
GB_16, GB_4, GB_GPU = 7509245952 / 2**30, 2.41, 6.0      # checkpoint bf16; alocado após carregar em 4 bits (medido); VRAM total

with safe_open(os.path.join(SNAP, "model-00001-of-00002.safetensors"), "pt") as f:
    w = f.get_tensor("model.layers.0.self_attn.q_proj.weight")[0, :64].float()
q, st = F.quantize_4bit(w.half().cuda(), quant_type="fp4", blocksize=64)
r = F.dequantize_4bit(q, st).float().cpu().numpy().ravel()
w = w.numpy(); escala = float(st.absmax[0]); niveis = np.array(sorted(set(round(float(x), 6) for x in F.get_4bit_type("fp4", device="cuda"))))
wn, rn = w / escala, r / escala
k = int(np.argmax(np.abs(wn - rn) * (np.abs(wn) > 0.2)))   # peso de exemplo: arredondamento bem visível

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), dpi=220, gridspec_kw={"width_ratios": [1.15, 1.6, 1.0], "wspace": 0.32})
fig.patch.set_facecolor(BG)
for a in ax:
    a.set_facecolor(BG); [a.spines[s].set_visible(False) for s in ("top", "right")]
    a.spines["left"].set_color(GRID); a.spines["bottom"].set_color(GRID); a.tick_params(colors=INK2, labelsize=8)

# (a) pesos originais do bloco
x = np.arange(64)
ax[0].vlines(x, 0, w, color=AZUL, lw=1.6); ax[0].axhline(0, color=GRID, lw=1)
i_max = int(np.argmax(np.abs(w)))
ax[0].plot(i_max, w[i_max], "o", ms=6, color=LARANJA, zorder=3)
ax[0].annotate(f"escala = maior |peso|\n= {escala:.4f} (guardada em 32 bits)", (i_max, w[i_max]), xytext=(-12, 2),
               textcoords="offset points", fontsize=8, color=INK, ha="right", va="center")
ax[0].set_ylim(-0.065, 0.068)
ax[0].set_title("1. Bloco de 64 pesos em 16 bits", fontsize=9.5, fontweight="bold", color=INK, loc="left")
ax[0].set_xlabel("posição no bloco", fontsize=8.5, color=INK2); ax[0].set_ylabel("peso", fontsize=8.5, color=INK2)

# (b) normalização e arredondamento para FP4
for v in niveis:
    ax[1].axvline(v, color=GRID, lw=1, zorder=0)
y = np.random.default_rng(0).uniform(0.15, 0.72, 64)
ax[1].scatter(wn, y, s=14, color=AZUL, zorder=2, label="peso / escala")
ax[1].scatter(rn, np.full(64, 0.02), s=22, marker="|", color=AZUL_ESC, zorder=2, label="valor FP4 guardado (4 bits)")
ax[1].annotate("", xy=(rn[k], 0.05), xytext=(wn[k], y[k]), arrowprops=dict(arrowstyle="-|>", color=LARANJA, lw=1.4), zorder=4)
ax[1].plot(wn[k], y[k], "o", ms=6, color=LARANJA, zorder=5)
ax[1].annotate(f"exemplo: {w[k]:+.4f} / escala = {wn[k]:+.3f}\n→ arredondado para {rn[k]:+.3f}", (wn[k], y[k]), xytext=(-1.05, 0.98),
               fontsize=8, color=INK, ha="left", va="bottom", arrowprops=dict(arrowstyle="-", color=LARANJA, lw=0.8))
ax[1].set_xlim(-1.08, 1.08); ax[1].set_ylim(-0.05, 1.3); ax[1].set_yticks([])
ax[1].set_xticks([-1, -0.5, 0, 0.5, 1])
ax[1].set_xlabel("valores representáveis em FP4 (linhas verticais)", fontsize=8.5, color=INK2)
ax[1].set_title("2. Dividir pela escala e arredondar para FP4", fontsize=9.5, fontweight="bold", color=INK, loc="left")
ax[1].legend(loc="upper right", fontsize=7.5, frameon=False)

# (c) memória do modelo
rot = ["16 bits", "4 bits"]; val = [GB_16, GB_4]
ax[2].barh([1, 0], val, height=0.5, color=[GRID, AZUL])
for yy, v in zip([1, 0], val):
    ax[2].text(v + 0.12, yy, f"{v:.1f} GB", va="center", fontsize=8.5, color=INK)
ax[2].axvline(GB_GPU, color=LARANJA, lw=1.4, ls="--"); ax[2].text(GB_GPU, 1.55, "6 GB da GPU", color=INK, fontsize=8, ha="center")
ax[2].set_yticks([1, 0]); ax[2].set_yticklabels(rot, fontsize=8.5, color=INK); ax[2].set_xlim(0, 8.4); ax[2].set_ylim(-0.5, 1.8)
ax[2].set_xlabel("memória do modelo (GB)", fontsize=8.5, color=INK2)
ax[2].set_title("3. Resultado no modelo", fontsize=9.5, fontweight="bold", color=INK, loc="left")

out = os.path.dirname(os.path.abspath(__file__))
for e in ("pdf", "png"): fig.savefig(os.path.join(out, f"fig_quantizacao.{e}"), bbox_inches="tight", facecolor=BG)
print(f"escala {escala:.4f} | exemplo w={w[k]:+.4f} -> n={wn[k]:+.3f} -> {rn[k]:+.3f} -> rec {r[k]:+.4f} | erro médio/escala {np.mean(np.abs(wn-rn)):.3f}")
