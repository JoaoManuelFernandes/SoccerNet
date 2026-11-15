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
W, H = 10.5, 3.2
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


y1, y2 = 0.8, 2.6                          # linha principal (partição em curso) e linha das alternativas
hd = 1.0                                   # altura dos losangos
P0, P1, D1, D2, PF = (0.05, 1.2), (1.45, 1.35), (3.0, 1.5), (6.6, 1.5), (9.25, 1.2)
B1, G1, B2, G2 = (3.1, 1.3), (4.75, 1.35), (6.7, 1.3), (8.55, 1.35)
passo(P0[0], y1, P0[1], ["Médias por", "tracklet"])
passo(P1[0], y1, P1[1], ["k-means na", "reidentificação"])
decisao(D1[0], y1, D1[1], hd, ["Separação", "clara?"])
decisao(D2[0], y1, D2[1], hd, ["Duas", "equipas?"])
passo(PF[0], y1, PF[1], ["Partição", "final"], lw=1.6)
passo(B1[0], y2, B1[1], ["k-means", "na cor"])
decisao(G1[0], y2, G1[1], hd, ["Cor", "aceite?"])
passo(B2[0], y2, B2[1], ["Resgate", "pela cor"])
decisao(G2[0], y2, G2[1], hd, ["Resgate", "aceite?"])


def ponto(x, y, r=0.045):                  # ponto de junção na linha principal
    forma(MSO_SHAPE.OVAL, x - r, y - r, 2 * r, 2 * r, INK2, INK2, 0.5)


def rot(x, y, t):
    texto(x, y - 0.1, 0.45, 0.2, t, 9, INK2)


cx = lambda D: D[0] + D[1] / 2
# linha principal
seta(P0[0] + P0[1], y1, P1[0], y1); seta(P1[0] + P1[1], y1, D1[0], y1)
seta(D1[0] + D1[1], y1, D2[0], y1); rot(D1[0] + D1[1] + 0.02, y1 - 0.18, "sim")
seta(D2[0] + D2[1], y1, PF[0], y1); rot(D2[0] + D2[1] + 0.02, y1 - 0.18, "sim")
# primeira alternativa: cor
seta(cx(D1), y1 + hd / 2, cx(D1), y2 - 0.36); rot(cx(D1) + 0.25, (y1 + y2) / 2 + 0.05, "não")
seta(B1[0] + B1[1], y2, G1[0], y2)
ponto(cx(G1), y1); seta(cx(G1), y2 - hd / 2, cx(G1), y1 + 0.05); rot(cx(G1) + 0.25, (y1 + y2) / 2 + 0.05, "não")
seta(G1[0] + G1[1], y2, D2[0] + D2[1] * 0.25, y1 + hd * 0.25); rot(G1[0] + G1[1] + 0.05, y2 - 0.32, "sim")
# segunda alternativa: resgate
seta(cx(D2), y1 + hd / 2, cx(D2), y2 - 0.36); rot(cx(D2) + 0.25, (y1 + y2) / 2 + 0.05, "não")
seta(B2[0] + B2[1], y2, G2[0], y2)
xm = 8.75; ponto(xm, y1); seta(cx(G2), y2 - hd / 2, xm + 0.02, y1 + 0.05); rot(cx(G2) - 0.05, (y1 + y2) / 2 + 0.05, "não")
seta(G2[0] + G2[1], y2, G2[0] + G2[1], y1 + 0.36); rot(G2[0] + G2[1] + 0.22, (y1 + y2) / 2 + 0.05, "sim")

prs.save(f"{AQ}/fig_auditoria.pptx"); print("-> fig_auditoria.pptx")
