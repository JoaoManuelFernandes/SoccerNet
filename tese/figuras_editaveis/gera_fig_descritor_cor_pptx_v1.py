"""Figura do descritor de cor (Secção 3.3, fig:descritor-cor) como PowerPoint editável, com o mesmo conteúdo de
sn-gamestate/docs/figuras_tese/fig_descritor_cor.py: recorte real (SNGS-116, imagem 1, caixa da anotação), região do tronco
em esquema (o mapa de atenção do BPBreID não é guardado), píxeis da banda central de recurso, histogramas HSV de 8 classes,
média Lab e vetor final. Imagens entram como imagens; barras, eixos, caixas e textos são formas editáveis.
Mesmas cores e letra das outras figuras editáveis (fig_jnr_janelas, fig_pipeline_modulos).
Uso: PYTHONPATH=pptx_libs venv_tl/bin/python gera_fig_descritor_cor_pptx.py -> fig_descritor_cor.pptx (depois pptx_para_pdf.py)"""
import os
import numpy as np, cv2
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

AQ = os.path.dirname(os.path.abspath(__file__)); IM = f"{AQ}/img_descritor_cor"; os.makedirs(IM, exist_ok=True)
AZUL, AZUL2, AZUL3, PALE = "2A78D6", "6EA0E3", "A9C6EF", "E9F1FB"
INK, INK2, GRID, BG = "0B0B0B", "52514E", "D8D8D4", "FCFCFB"

# ---- dados: os mesmos cálculos do script matplotlib (banda central 25%-75%, modo de recurso do código)
img = cv2.imread("/home/joao/soccernet/data/SoccerNetGS/test/SNGS-116/img1/000001.jpg")
x0, y0, cw, ch = 478, 565, 51, 140
crop = img[y0:y0 + ch, x0:x0 + cw]; hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV); lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
mask = np.zeros((ch, cw), np.uint8); mask[int(0.25 * ch):int(0.75 * ch), :] = 1
hists = []
for c, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
    hh = cv2.calcHist([hsv], [c], mask, [8], [lo, hi]).flatten(); hists.append(hh / hh.sum())
hist = np.concatenate(hists); lab_m = lab[mask > 0].mean(0) / 255.0
sel = crop.copy(); sel[mask == 0] = (232, 235, 235)
cv2.imwrite(f"{IM}/recorte.png", crop); cv2.imwrite(f"{IM}/selecionados.png", sel)

# ---- diapositivo
W, H = 10.3, 4.6
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
sl = prs.slides.add_slide(prs.slide_layouts[6]); sl.background.fill.solid(); sl.background.fill.fore_color.rgb = RGBColor.from_string(BG)
S = sl.shapes


def caixa(x, y, w, h, fill, borda=None, lw=1.0, forma="rect", alfa=None, nome=None):
    s = S.add_shape({"rect": MSO_SHAPE.RECTANGLE, "round": MSO_SHAPE.ROUNDED_RECTANGLE, "oval": MSO_SHAPE.OVAL}[forma],
                    Inches(x), Inches(y), Inches(w), Inches(h))
    if forma == "round": s.adjustments[0] = 0.12
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(fill)
    if alfa is not None:                                   # transparência: a:alpha dentro da cor de preenchimento
        etree.SubElement(s.fill._xPr.find(qn("a:solidFill")).find(qn("a:srgbClr")), qn("a:alpha"), val=str(int(alfa * 100000)))
    if borda: s.line.color.rgb = RGBColor.from_string(borda); s.line.width = Pt(lw)
    else: s.line.fill.background()
    etree.SubElement(s._element.spPr, qn("a:effectLst"))
    if nome: s.name = nome
    return s


def texto(x, y, w, h, txt, pt, cor=INK, bold=False, mono=False, alinh="l", ancora="c"):
    """uma linha por caixa de texto (o conversor para PDF só desenha uma linha por caixa)"""
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.word_wrap = False
    tf.vertical_anchor = {"c": MSO_ANCHOR.MIDDLE, "t": MSO_ANCHOR.TOP, "b": MSO_ANCHOR.BOTTOM}[ancora]
    p = tf.paragraphs[0]; p.alignment = {"c": PP_ALIGN.CENTER, "l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT}[alinh]
    r = p.add_run(); r.text = txt; f = r.font; f.size = Pt(pt); f.bold = bold
    f.name = "Courier New" if mono else "Arial"; f.color.rgb = RGBColor.from_string(cor)
    return tb


def linha(x1, y1, x2, y2, cor=GRID, lw=0.75, seta=False):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(cor); c.line.width = Pt(lw)
    if seta: etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle", w="med", len="med")
    return c


def imagem(path, x, y, w, h, alfa=None):
    p = S.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    if alfa is not None: etree.SubElement(p._element.find(".//" + qn("a:blip")), qn("a:alphaModFix"), amt=str(int(alfa * 100000)))
    return p


def titulo(x, y, l1, l2=None):
    texto(x, y, 2.0, 0.24, l1, 10.5, INK, bold=True)
    if l2: texto(x, y + 0.24, 2.0, 0.2, l2, 9, INK2)


# ---- painéis 1 a 3: imagens (altura 3.3 pol., proporção do recorte)
ih = 3.3; iw = ih * cw / ch; iy = 0.95
for k, (l1, l2) in enumerate((("1. Recorte", "da deteção"), ("2. Região do tronco", "em esquema"), ("3. Píxeis", "selecionados"))):
    titulo(0.15 + k * 1.85, 0.12, l1, l2)
imagem(f"{IM}/recorte.png", 0.15, iy, iw, ih)
imagem(f"{IM}/recorte.png", 2.0, iy, iw, ih, alfa=0.35)
caixa(2.0 + 0.075 * iw, iy + 0.19 * ih, 0.85 * iw, 0.42 * ih, AZUL, forma="oval", alfa=0.55, nome="tronco")
texto(2.0, iy + 0.40 * ih - 0.12, iw, 0.24, "tronco", 10, "FFFFFF", bold=True, alinh="c")
imagem(f"{IM}/selecionados.png", 3.85, iy, iw, ih)
for j, t in enumerate(("atenção ≥ 0.25 do máximo;", "recurso: banda 25%–75%")):
    texto(3.85 - 0.2, iy + ih - 0.62 + j * 0.2, iw + 0.4, 0.2, t, 8, INK2, alinh="c")

# ---- painel 4: histogramas HSV (barras como retângulos)
hx0, hx1, hy0, hy1, vmax = 5.75, 8.3, 0.75, 2.35, 0.40          # área do gráfico; topo = vmax
titulo(5.4, 0.12, "4. Histogramas HSV", "8 classes por canal, 24 valores")
for v in (0.0, 0.1, 0.2, 0.3, 0.4):
    yy = hy1 - v / vmax * (hy1 - hy0); linha(hx0, yy, hx1, yy, GRID if v else INK2, 0.75 if v else 1.0)
    texto(hx0 - 0.42, yy - 0.09, 0.36, 0.18, f"{v:.1f}", 8, INK2, alinh="r")
slot = (hx1 - hx0) / 24
for i, v in enumerate(hist):
    bh = max(0.004, v / vmax * (hy1 - hy0))
    caixa(hx0 + i * slot + 0.12 * slot, hy1 - bh, 0.76 * slot, bh, (AZUL, AZUL2, AZUL3)[i // 8], nome=f"barra_{'HSV'[i // 8]}{i % 8 + 1}")
for g, nm in enumerate(("H (matiz)", "S (saturação)", "V (valor)")):
    if g: linha(hx0 + 8 * g * slot, hy0, hx0 + 8 * g * slot, hy1, GRID, 0.75)
    texto(hx0 + 8 * g * slot, hy1 + 0.06, 8 * slot, 0.2, nm, 8.5, INK2, alinh="c")

# ---- painel 5: média Lab (barras horizontais)
lx0, lx1, ly0 = 5.75, 8.3, 3.25
titulo(5.4, 2.72, "5. Média Lab", "mesmos píxeis, de 0 a 1, 3 valores")
for v in (0.0, 0.5, 1.0):
    xx = lx0 + v * (lx1 - lx0); linha(xx, ly0 - 0.05, xx, ly0 + 1.05, GRID if v else INK2, 0.75 if v else 1.0)
    texto(xx - 0.2, ly0 + 1.08, 0.4, 0.18, f"{v:.1f}", 8, INK2, alinh="c")
for j, (nm, v) in enumerate(zip("Lab", lab_m)):
    yy = ly0 + 0.05 + j * 0.33
    caixa(lx0, yy, v * (lx1 - lx0), 0.24, AZUL, nome=f"lab_{nm}"); texto(lx0 - 0.3, yy, 0.22, 0.24, nm, 9, INK2, alinh="r")

# ---- painel 6: vetor final
vx, vw = 8.6, 1.55
titulo(8.6, 0.12, "6. Vetor final", "por deteção")
caixa(vx, 1.0, vw, 0.75, "FFFFFF", INK2, 1.0, forma="round")
texto(vx, 1.12, vw, 0.24, "embeddings", 9, INK, mono=True, alinh="c")
texto(vx, 1.38, vw, 0.24, "ReID, 256 d (inalterado)", 8.5, INK2, alinh="c")
linha(vx + vw / 2, 1.77, vx + vw / 2, 2.13, INK2, 1.2, seta=True)
caixa(vx, 2.15, vw, 1.05, PALE, AZUL, 1.4, forma="round")
texto(vx, 2.25, vw, 0.24, "embeddings_with_colors", 7.5, INK, mono=True, alinh="c")
texto(vx, 2.53, vw, 0.24, "256 ReID + 27 cor", 8.5, INK, alinh="c")
texto(vx, 2.81, vw, 0.24, "= 283 d (coluna nova)", 8.5, INK, alinh="c")
for j, t in enumerate(("o agrupamento em equipas", "usa só os 27 valores de cor")):
    texto(vx - 0.1, 3.45 + j * 0.2, vw + 0.2, 0.2, t, 8, INK2, alinh="c")

prs.save(f"{AQ}/fig_descritor_cor.pptx"); print("-> fig_descritor_cor.pptx | Lab", lab_m.round(3), "| max hist", hist.max().round(3))
