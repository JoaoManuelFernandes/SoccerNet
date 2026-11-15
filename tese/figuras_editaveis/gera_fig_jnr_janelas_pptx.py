"""Figura da secção 3.2.3 (confiança da leitura e decisão por janela) como PowerPoint editável: recortes do tracklet em janelas de 6,
os 5 mais nítidos de cada janela vão ao Qwen numa só chamada, a resposta tem a probabilidade de cada dígito, a confiança é o produto,
só votam as janelas com confiança >= 0.5 e o tracklet fica com o número mais votado. Mesmas cores e letra da Figura 3.1.
Uso: PYTHONPATH=pptx_libs python3 gera_fig_jnr_janelas_pptx.py -> fig_jnr_janelas.pptx (depois pptx_para_pdf.py para PDF/PNG)"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

AZUL, PALE, INK, INK2, BG = "2A78D6", "E9F1FB", "0B0B0B", "52514E", "FCFCFB"
VERDE, VERDE_P, CINZA, CINZA_P, BORDA = "2E7D4F", "E4F2E9", "8A8D91", "EEEFF0", "C9CCD1"
W, H = 10.0, 4.55
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
sl = prs.slides.add_slide(prs.slide_layouts[6]); sl.background.fill.solid(); sl.background.fill.fore_color.rgb = RGBColor.from_string(BG)
S = sl.shapes


def caixa(x, y, w, h, fill, borda=None, lw=1.0, redonda=True, nome=None):
    s = S.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if redonda else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if redonda: s.adjustments[0] = 0.18
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(fill)
    if borda: s.line.color.rgb = RGBColor.from_string(borda); s.line.width = Pt(lw)
    else: s.line.fill.background()
    etree.SubElement(s._element.spPr, qn("a:effectLst"))
    if nome: s.name = nome
    return s


def texto(x, y, w, h, txt, pt, cor=INK, bold=False, mono=False, alinh="c", ancora="c"):
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.word_wrap = False
    tf.vertical_anchor = {"c": MSO_ANCHOR.MIDDLE, "t": MSO_ANCHOR.TOP, "b": MSO_ANCHOR.BOTTOM}[ancora]
    p = tf.paragraphs[0]; p.alignment = {"c": PP_ALIGN.CENTER, "l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT}[alinh]
    r = p.add_run(); r.text = txt; f = r.font; f.size = Pt(pt); f.bold = bold
    f.name = "Courier New" if mono else "Arial"; f.color.rgb = RGBColor.from_string(cor)
    return tb


def seta(x1, y1, x2, y2, cor=CINZA, lw=1.4):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(cor); c.line.width = Pt(lw)
    etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle", w="med", len="med")
    return c


# títulos das colunas (cada linha de texto é uma caixa: o conversor para PDF só desenha uma linha por caixa)
COL = dict(jan=(0.15, 3.55), leit=(3.95, 1.3), conf=(5.45, 2.0), lim=(7.6, 0.95), dec=(8.75, 1.1))
for k, t in (("jan", "Janelas de 6 recortes"), ("leit", "Leitura"), ("conf", "Confiança"), ("lim", "Limiar"), ("dec", "Decisão")):
    texto(COL[k][0], 0.08, COL[k][1], 0.3, t, 12, INK, bold=True)
for k, l1, l2 in (("jan", "do tracklet, por ordem temporal", "× = recorte menos nítido, fica de fora"),
                  ("leit", "uma chamada", "com 5 recortes"), ("conf", "produto das probabilidades", "dos dígitos"),
                  ("lim", "vota se ≥ 0.5", ""), ("dec", "mais votado", "")):
    texto(COL[k][0], 0.36, COL[k][1], 0.22, l1, 10, INK2)
    if l2: texto(COL[k][0], 0.56, COL[k][1], 0.22, l2, 10, INK2)

linhas = [  # (rótulo, posição do recorte descartado, resposta, cálculo, confiança, vota)
    ("Janela 1", 2, "«14»", "0.97 × 0.93", "0.90", True),
    ("Janela 2", 5, "«14»", "0.95 × 0.88", "0.84", True),
    ("Janela 3", 0, "«4»", "0.31", "0.31", False),
    ("Janela 4", 3, "recusa", "", "", False),
]
y0, dy, q = 0.95, 0.88, 0.42
ys = []
for k, (rot, desc, resp, calc, conf, vota) in enumerate(linhas):
    y = y0 + k * dy; ym = y + q / 2; ys.append((ym, vota))
    texto(0.15, y, 0.75, q, rot, 10, INK2, alinh="l")
    for j in range(6):
        x = 0.9 + j * (q + 0.05)
        if j == desc: caixa(x, y, q, q, CINZA_P, BORDA, 0.8, redonda=False); texto(x, y, q, q, "×", 12, CINZA)
        else: caixa(x, y, q, q, PALE, AZUL, 0.8, redonda=False); texto(x, y, q, q, "14" if resp != "recusa" and k < 2 else "?", 10, AZUL, bold=True)
    seta(3.8, ym, 3.97, ym)
    caixa(4.0, y - 0.02, 1.2, q + 0.04, PALE, AZUL, 1.0); texto(4.0, y - 0.02, 1.2, q + 0.04, "Qwen2.5-VL", 11, INK, bold=True)
    seta(5.23, ym, 5.42, ym)
    texto(5.45, y - 0.03, 0.75, q / 2 + 0.03, resp, 11, INK, bold=True, alinh="l")
    if calc: texto(5.45, ym, 2.6, q / 2, f"{calc} = {conf}" if calc != conf else f"{conf}", 11, INK2, mono=True, alinh="l")
    else: texto(5.45, ym, 2.6, q / 2, "sem número", 11, INK2, alinh="l")
    if vota: caixa(7.65, y + 0.03, 0.85, q - 0.06, VERDE_P, VERDE, 1.0); texto(7.65, y + 0.03, 0.85, q - 0.06, "vota", 11, VERDE, bold=True)
    else: caixa(7.65, y + 0.03, 0.85, q - 0.06, CINZA_P, CINZA, 1.0); texto(7.65, y + 0.03, 0.85, q - 0.06, "não vota", 10, CINZA, bold=True)
# decisão
yb, hb = y0 + 0.2, 2.1
caixa(8.8, yb, 1.0, hb, VERDE_P, VERDE, 1.2)
texto(8.8, yb + 0.15, 1.0, 0.25, "14: 2 votos", 10, INK2)
texto(8.8, yb + 0.42, 1.0, 0.25, "4: 0 votos", 10, INK2)
texto(8.8, yb + 0.8, 1.0, 0.6, "14", 30, VERDE, bold=True)
texto(8.8, yb + 1.5, 1.0, 0.25, "número do", 10, INK2)
texto(8.8, yb + 1.73, 1.0, 0.25, "tracklet", 10, INK2)
for ym, vota in ys:
    if vota: seta(8.52, ym, 8.78, min(max(ym, yb + 0.2), yb + hb - 0.2), VERDE)
prs.save("fig_jnr_janelas.pptx"); print("-> fig_jnr_janelas.pptx")
