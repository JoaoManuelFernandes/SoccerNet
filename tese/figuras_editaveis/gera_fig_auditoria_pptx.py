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
W, H = 5.6, 6.6
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


# fluxograma vertical: coluna principal à esquerda, alternativas à direita, só ligações horizontais e verticais
xs, xb = 1.35, 4.4                         # centro da coluna principal e da coluna das alternativas
wb, wd, hb, hd = 2.0, 1.9, 0.52, 0.8       # largura das caixas e dos losangos, alturas


def caixa_c(xc, yc, linhas, lw=1.0): passo(xc - wb / 2, yc, wb, linhas, h=hb, lw=lw)
def los_c(xc, yc, linhas): decisao(xc - wd / 2, yc, wd, hd, linhas)
def ponto(x, y, r=0.045): forma(MSO_SHAPE.OVAL, x - r, y - r, 2 * r, 2 * r, INK2, INK2, 0.5)
def rot(x, y, t): texto(x - 0.25, y - 0.1, 0.5, 0.2, t, 9, INK2)


y = dict(km=0.4, d1=1.3, g1=2.15, u1=2.95, d2=3.8, g2=4.65, u2=5.45, fim=6.25)
passo(xs - 1.2, y["km"], 2.4, ["k-means na reidentificação", "(médias por tracklet)"], h=hb)   # mais larga: o texto é maior
los_c(xs, y["d1"], ["Separação", "clara?"])
caixa_c(xb, y["d1"], ["k-means na cor"])
los_c(xb, y["g1"], ["Cor", "aceite?"])
caixa_c(xb, y["u1"], ["usa a partição", "por cor"])
los_c(xs, y["d2"], ["Duas", "equipas?"])
caixa_c(xb, y["d2"], ["Resgate pela cor"])
los_c(xb, y["g2"], ["Resgate", "aceite?"])
caixa_c(xb, y["u2"], ["usa a partição", "de resgate"])
caixa_c(xs, y["fim"], ["Partição final"], lw=1.6)

seta(xs, y["km"] + hb / 2, xs, y["d1"] - hd / 2)
for d, g, u in (("d1", "g1", "u1"), ("d2", "g2", "u2")):
    seta(xs + wd / 2, y[d], xb - wb / 2, y[d]); rot((xs + wd / 2 + xb - wb / 2) / 2, y[d] - 0.15, "não")
    seta(xb, y[d] + hb / 2, xb, y[g] - hd / 2)
    seta(xb, y[g] + hd / 2, xb, y[u] - hb / 2); rot(xb + 0.25, (y[g] + hd / 2 + y[u] - hb / 2) / 2, "sim")
    seta(xb - wd / 2, y[g], xs, y[g]); rot((xs + xb - wd / 2) / 2, y[g] - 0.15, "não"); ponto(xs, y[g])
    seta(xb - wb / 2, y[u], xs, y[u]); ponto(xs, y[u])
seta(xs, y["d1"] + hd / 2, xs, y["d2"] - hd / 2); rot(xs - 0.25, y["d1"] + hd / 2 + 0.17, "sim")
seta(xs, y["d2"] + hd / 2, xs, y["fim"] - hb / 2); rot(xs - 0.25, y["d2"] + hd / 2 + 0.17, "sim")

prs.save(f"{AQ}/fig_auditoria.pptx"); print("-> fig_auditoria.pptx")
