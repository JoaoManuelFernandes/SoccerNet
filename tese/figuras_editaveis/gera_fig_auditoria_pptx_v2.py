"""Figura da auditoria do agrupamento em equipas (Secção 3.3.2, fig:auditoria) como PowerPoint editável, em fluxograma
simples (12/10: a versão anterior, com frases e contagens nas caixas, «parecia IA»; está em gera_fig_auditoria_pptx_v1.py).
Retângulos = passos, losangos = decisões. Sem desvios, a partição é a da reidentificação (como na baseline); o gatilho A
(silhueta < 0.28) ou B (>= 85% numa metade e centros < 2.04) leva ao k-means na cor (27 números); o grupo menor com < 8% das
deteções leva ao resgate pela cor medida nas imagens. As exigências de 20% e a da arbitragem ficam no texto.
Uso: PYTHONPATH=pptx_libs venv_tl/bin/python gera_fig_auditoria_pptx.py -> fig_auditoria.pptx (depois pptx_para_pdf.py)"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

AQ = os.path.dirname(os.path.abspath(__file__))
AZUL, PALE, INK, INK2, BG = "2A78D6", "E9F1FB", "0B0B0B", "52514E", "FCFCFB"
W, H = 10.0, 3.35
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
sl = prs.slides.add_slide(prs.slide_layouts[6]); sl.background.fill.solid(); sl.background.fill.fore_color.rgb = RGBColor.from_string(BG)
S = sl.shapes


def forma(tipo, x, y, w, h, fill, borda, lw=1.0):
    s = S.add_shape(tipo, Inches(x), Inches(y), Inches(w), Inches(h))
    if tipo == MSO_SHAPE.ROUNDED_RECTANGLE: s.adjustments[0] = 0.1
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(fill)
    s.line.color.rgb = RGBColor.from_string(borda); s.line.width = Pt(lw)
    etree.SubElement(s._element.spPr, qn("a:effectLst"))
    return s


def texto(x, y, w, h, txt, pt, cor=INK, bold=False):
    """uma linha por caixa de texto (o conversor para PDF só desenha uma linha por caixa)"""
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = txt; f = r.font; f.size = Pt(pt); f.bold = bold; f.name = "Arial"; f.color.rgb = RGBColor.from_string(cor)
    return tb


def seta(x1, y1, x2, y2):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(INK2); c.line.width = Pt(1.2)
    etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle", w="med", len="med")


def passo(x, yc, w, linhas, h=0.72, lw=1.0):
    forma(MSO_SHAPE.ROUNDED_RECTANGLE, x, yc - h / 2, w, h, PALE, AZUL, lw)
    for k, t in enumerate(linhas):
        texto(x, yc - 0.11 * len(linhas) + 0.22 * k, w, 0.22, t, 10, INK)


def decisao(x, yc, w, h, linhas):
    forma(MSO_SHAPE.DIAMOND, x, yc - h / 2, w, h, "FFFFFF", INK2)
    for k, t in enumerate(linhas):
        texto(x, yc - 0.11 * len(linhas) + 0.22 * k, w, 0.22, t, 10, INK)


y1, y2 = 0.85, 2.55                        # linha principal e linha dos desvios
P0, P1, D1, D2, PF = (0.1, 1.45), (1.85, 1.45), (3.65, 1.8), (6.0, 1.8), (8.35, 1.5)
hd = 1.2                                   # altura dos losangos
passo(P0[0], y1, P0[1], ["Médias por", "tracklet"])
passo(P1[0], y1, P1[1], ["k-means na", "reidentificação"])
decisao(D1[0], y1, D1[1], hd, ["Gatilho", "A ou B?"])
decisao(D2[0], y1, D2[1], hd, ["Grupo menor", "< 8%?"])
passo(PF[0], y1, PF[1], ["Partição", "final"], lw=1.6)
B1 = (D1[0] - 0.05, D1[1] + 0.1); B2 = (D2[0] - 0.05, D2[1] + 0.1)
passo(B1[0], y2, B1[1], ["k-means na cor", "(27 números)"])
passo(B2[0], y2, B2[1], ["Resgate pela cor", "das imagens"])

seta(P0[0] + P0[1], y1, P1[0], y1); seta(P1[0] + P1[1], y1, D1[0], y1)
seta(D1[0] + D1[1], y1, D2[0], y1); seta(D2[0] + D2[1], y1, PF[0], y1)
texto(D1[0] + D1[1], y1 - 0.25, D2[0] - D1[0] - D1[1], 0.2, "não", 9, INK2)
texto(D2[0] + D2[1], y1 - 0.25, PF[0] - D2[0] - D2[1], 0.2, "não", 9, INK2)
for D in (D1, D2):                         # «sim» desce do vértice inferior
    xc = D[0] + D[1] / 2; seta(xc, y1 + hd / 2, xc, y2 - 0.36); texto(xc + 0.08, (y1 + hd / 2 + y2 - 0.36) / 2 - 0.1, 0.4, 0.2, "sim", 9, INK2)
# os desvios voltam à linha principal: a cor vai à segunda verificação, o resgate à partição final
seta(B1[0] + B1[1], y2 - 0.15, D2[0] + D2[1] * 0.25, y1 + hd * 0.25)
seta(B2[0] + B2[1], y2 - 0.15, PF[0] + 0.35, y1 + 0.36)

prs.save(f"{AQ}/fig_auditoria.pptx"); print("-> fig_auditoria.pptx")
