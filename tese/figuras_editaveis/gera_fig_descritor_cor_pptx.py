"""Figura do descritor de cor (Secção 3.3, fig:descritor-cor) como PowerPoint editável, versão simplificada (12/10, pedido do João:
«a imagem tem muita informação»). Quatro passos: recorte real (SNGS-116, imagem 1, caixa da anotação) -> píxeis do tronco
(a região do tronco é aproximada por uma elipse, porque o mapa de atenção do BPBreID não é guardado; os valores da figura são
calculados sobre os píxeis dentro da elipse) -> cor do tronco (3 histogramas HSV de 8 intervalos + cor média Lab = 27 valores)
-> vetor final (coluna nova com 283 valores; a original fica intacta). Versão anterior, com 6 painéis: gera_fig_descritor_cor_pptx_v1.py.
Imagens entram como imagens; barras, caixas, setas e textos são formas editáveis (uma linha de texto por caixa).
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
INK, INK2, CINZA, BG = "0B0B0B", "52514E", "8A8D91", "FCFCFB"

# ---- dados: píxeis dentro da elipse do tronco (esquema), histogramas de 8 intervalos por canal HSV e média Lab
img = cv2.imread("/home/joao/soccernet/data/SoccerNetGS/test/SNGS-116/img1/000001.jpg")
x0, y0, cw, ch = 478, 565, 51, 140
crop = img[y0:y0 + ch, x0:x0 + cw]; hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV); lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
EC, EA = (cw / 2, 0.355 * ch), (0.36 * cw, 0.165 * ch)            # centro e semieixos da elipse, em píxeis do recorte
yy_, xx_ = np.mgrid[0:ch, 0:cw] + 0.5                                  # centro de cada píxel: a máscara coincide com a elipse desenhada
mask = ((((xx_ - EC[0]) / EA[0]) ** 2 + ((yy_ - EC[1]) / EA[1]) ** 2) <= 1).astype(np.uint8)
hists = []
for c, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
    hh = cv2.calcHist([hsv], [c], mask, [8], [lo, hi]).flatten(); hists.append(hh / hh.sum())
lab_m = lab[mask > 0].mean(0)
media_bgr = cv2.cvtColor(np.uint8([[lab_m]]), cv2.COLOR_Lab2BGR)[0, 0]; MEDIA = "%02X%02X%02X" % tuple(int(v) for v in media_bgr[::-1])
fundo = np.full_like(crop, (251, 252, 252)); tronco = np.where(mask[..., None] > 0, crop, (0.3 * crop + 0.7 * fundo).astype(np.uint8))
cv2.imwrite(f"{IM}/recorte.png", crop); cv2.imwrite(f"{IM}/tronco.png", tronco)

# ---- diapositivo
W, H = 10.0, 4.15
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
sl = prs.slides.add_slide(prs.slide_layouts[6]); sl.background.fill.solid(); sl.background.fill.fore_color.rgb = RGBColor.from_string(BG)
S = sl.shapes


def caixa(x, y, w, h, fill, borda=None, lw=1.0, forma="rect", alfa=None, nome=None):
    s = S.add_shape({"rect": MSO_SHAPE.RECTANGLE, "round": MSO_SHAPE.ROUNDED_RECTANGLE, "oval": MSO_SHAPE.OVAL}[forma],
                    Inches(x), Inches(y), Inches(w), Inches(h))
    if forma == "round": s.adjustments[0] = 0.12
    if fill:
        s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(fill)
        if alfa is not None: etree.SubElement(s.fill._xPr.find(qn("a:solidFill")).find(qn("a:srgbClr")), qn("a:alpha"), val=str(int(alfa * 100000)))
    else: s.fill.background()
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


def linha(x1, y1, x2, y2, cor=INK2, lw=1.0, seta=False):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(cor); c.line.width = Pt(lw)
    if seta: etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle", w="med", len="med")
    return c


def titulo(x, y, w, l1, l2=None):
    texto(x, y, w, 0.26, l1, 11, INK, bold=True, alinh="c")
    if l2: texto(x, y + 0.27, w, 0.22, l2, 9.5, INK2, alinh="c")


# colunas (x, largura) e altura útil
C1, C2, C3, C4 = (0.15, 1.6), (2.15, 1.6), (4.25, 3.1), (7.85, 2.0)
iy, ih = 0.85, 3.1; iw = ih * cw / ch
titulo(C1[0], 0.1, C1[1], "1. Recorte", "da deteção")
titulo(C2[0], 0.1, C2[1], "2. Tronco", "aproximado por uma elipse")
titulo(C3[0], 0.1, C3[1], "3. Cor do tronco", "27 valores")
titulo(C4[0], 0.1, C4[1], "4. Vetor final", "por deteção")

# 1 e 2: imagens
p1 = C1[0] + (C1[1] - iw) / 2; p2 = C2[0] + (C2[1] - iw) / 2
S.add_picture(f"{IM}/recorte.png", Inches(p1), Inches(iy), Inches(iw), Inches(ih))
S.add_picture(f"{IM}/tronco.png", Inches(p2), Inches(iy), Inches(iw), Inches(ih))
sx = iw / cw
caixa(p2 + (EC[0] - EA[0]) * sx, iy + (EC[1] - EA[1]) * sx, 2 * EA[0] * sx, 2 * EA[1] * sx, None, AZUL, 1.5, forma="oval", nome="elipse_tronco")
ym = iy + ih / 2
linha(C1[0] + C1[1] - 0.12, ym, C2[0] + 0.12, ym, seta=True)
linha(C2[0] + C2[1] - 0.12, ym, C3[0] + 0.05, ym, seta=True)

# 3: três histogramas HSV (8 intervalos cada) e cor média Lab
gx0, gw, gap, gy1, gh = C3[0] + 0.1, 0.86, 0.12, iy + 1.55, 1.3                 # base dos gráficos em gy1, altura máxima gh
vmax = max(h.max() for h in hists)
for g, (nm, cor) in enumerate((("matiz (H)", AZUL), ("saturação (S)", AZUL2), ("valor (V)", AZUL3))):
    bx = gx0 + g * (gw + gap); slot = gw / 8
    for i, v in enumerate(hists[g]):
        bh = max(0.006, v / vmax * gh); caixa(bx + i * slot + 0.1 * slot, gy1 - bh, 0.8 * slot, bh, cor, nome=f"barra_{'HSV'[g]}{i + 1}")
    linha(bx, gy1, bx + gw, gy1, INK2, 1.0)
    texto(bx - 0.1, gy1 + 0.06, gw + 0.2, 0.2, nm, 9, INK2, alinh="c")
texto(gx0, gy1 + 0.32, 3 * gw + 2 * gap, 0.2, "histogramas HSV, 3 × 8 intervalos = 24 valores", 9, INK, alinh="c")
sy = gy1 + 0.75
caixa(gx0 + 0.55, sy, 0.42, 0.42, MEDIA, INK2, 0.75, nome="cor_media")
texto(gx0 + 1.07, sy + 0.01, 1.9, 0.2, "cor média (Lab)", 9, INK, alinh="l")
texto(gx0 + 1.07, sy + 0.21, 1.9, 0.2, "3 valores", 9, INK2, alinh="l")
linha(C3[0] + C3[1] - 0.05, ym, C4[0] + 0.08, ym, seta=True)

# 4: vetor final
vx, vw = C4[0] + 0.1, C4[1] - 0.15
caixa(vx, iy + 0.35, vw, 0.8, "FFFFFF", INK2, 1.0, forma="round")
texto(vx, iy + 0.47, vw, 0.24, "embeddings", 9.5, INK, mono=True, alinh="c")
texto(vx, iy + 0.75, vw, 0.24, "256 valores, inalterado", 9, INK2, alinh="c")
caixa(vx, iy + 1.55, vw, 1.15, PALE, AZUL, 1.4, forma="round")
texto(vx, iy + 1.67, vw, 0.24, "embeddings_with_colors", 8, INK, mono=True, alinh="c")
texto(vx, iy + 1.97, vw, 0.24, "256 + 27 = 283 valores", 9, INK, alinh="c")
texto(vx, iy + 2.25, vw, 0.24, "coluna nova", 9, INK2, alinh="c")
linha(vx + vw / 2, iy + 1.17, vx + vw / 2, iy + 1.53, seta=True)

prs.save(f"{AQ}/fig_descritor_cor.pptx"); print("-> fig_descritor_cor.pptx | cor média", MEDIA, "| píxeis na elipse", int(mask.sum()))
