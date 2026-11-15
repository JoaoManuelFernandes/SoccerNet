"""Figura da auditoria do agrupamento em equipas (Secção 3.3.2, fig:auditoria) como PowerPoint editável.
Caminho de cima = o da baseline (médias por tracklet -> k-means na reidentificação -> partição). As duas verificações
desviam para os ramos de baixo: separação pouco clara (gatilho A: silhueta < 0.28; B: >= 85% numa metade e centros < 2.04)
-> k-means nos 27 números de cor (aceite se o grupo menor >= 20%, só no gatilho A); resultado implausível (grupo menor
< 8% das deteções) -> resgate pela cor medida nas imagens (aceite se >= 20% em tracklets e deteções e arbitragem <= 25%).
Valores conferidos em sn_gamestate/team/tracklet_team_clustering_api.py; frequências do texto da tese (treino 49/57, cor 8/57,
plausibilidade 11/159, resgate aceite 8). Mesmas cores e letra das outras figuras editáveis.
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
VERDE, VERDE_P, CINZA, CINZA_P, LARANJA, LARANJA_P = "2E7D4F", "E4F2E9", "8A8D91", "F4F5F6", "C2571A", "FBEDE4"
W, H = 10.0, 4.25
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
sl = prs.slides.add_slide(prs.slide_layouts[6]); sl.background.fill.solid(); sl.background.fill.fore_color.rgb = RGBColor.from_string(BG)
S = sl.shapes


def caixa(x, y, w, h, fill, borda=None, lw=1.0, nome=None):
    s = S.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); s.adjustments[0] = 0.12
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(fill)
    if borda: s.line.color.rgb = RGBColor.from_string(borda); s.line.width = Pt(lw)
    else: s.line.fill.background()
    etree.SubElement(s._element.spPr, qn("a:effectLst"))
    if nome: s.name = nome
    return s


def texto(x, y, w, h, txt, pt, cor=INK, bold=False, alinh="c", ancora="c"):
    """uma linha por caixa de texto (o conversor para PDF só desenha uma linha por caixa)"""
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.word_wrap = False
    tf.vertical_anchor = {"c": MSO_ANCHOR.MIDDLE, "t": MSO_ANCHOR.TOP, "b": MSO_ANCHOR.BOTTOM}[ancora]
    p = tf.paragraphs[0]; p.alignment = {"c": PP_ALIGN.CENTER, "l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT}[alinh]
    r = p.add_run(); r.text = txt; f = r.font; f.size = Pt(pt); f.bold = bold; f.name = "Arial"; f.color.rgb = RGBColor.from_string(cor)
    return tb


def seta(x1, y1, x2, y2, cor=INK2, lw=1.3):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(cor); c.line.width = Pt(lw)
    etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle", w="med", len="med")
    return c


def bloco(x, y, w, h, linhas, fill, borda, cor_tit=INK, lw=1.0):
    """caixa com um título (negrito) e linhas de pormenor, centradas na vertical"""
    caixa(x, y, w, h, fill, borda, lw)
    alt = [0.26] + [0.2] * (len(linhas) - 1); y0 = y + (h - sum(alt)) / 2
    for k, (t, a) in enumerate(zip(linhas, alt)):
        texto(x, y0, w, a, t, 10 if k == 0 else 8.5, cor_tit if k == 0 else INK2, bold=k == 0); y0 += a


# rótulos das duas faixas
texto(0.1, 0.12, 6, 0.24, "Caminho da baseline, seguido na maioria dos vídeos", 9.5, INK2, bold=True, alinh="l")
texto(0.1, 3.92, 6, 0.24, "Desvios da auditoria, só quando uma verificação falha", 9.5, INK2, bold=True, alinh="l")

ys, hs = 0.55, 1.15          # faixa de cima
yb, hb = 2.6, 1.2            # faixa de baixo
C0, C1, Q1, Q2, F = (0.1, 1.3), (1.6, 1.3), (3.15, 2.1), (5.75, 2.1), (8.3, 1.6)
bloco(*C0[:1], ys, C0[1], hs, ["Médias", "por tracklet:", "reidentificação,", "cor e posição"], "FFFFFF", CINZA)
bloco(C1[0], ys, C1[1], hs, ["1. k-means", "na reidentificação", "dois grupos,", "10 inícios"], PALE, AZUL)
bloco(Q1[0], ys, Q1[1], hs, ["2. Separação clara?", "não, se A: silhueta < 0.28", "ou B: jogo numa metade", "e centros a < 2.04"], "FFFFFF", AZUL, lw=1.4)
bloco(Q2[0], ys, Q2[1], hs, ["3. Há duas equipas?", "não, se o grupo menor", "tiver < 8% das deteções", "de jogadores"], "FFFFFF", AZUL, lw=1.4)
bloco(F[0], ys, F[1], hs, ["Partição final", "duas equipas,", "segue para o lado", "do campo (3.3.3)"], VERDE_P, VERDE, VERDE, lw=1.4)
bloco(Q1[0], yb, Q1[1], hb, ["Agrupar pela cor", "k-means nos 27 números", "do descritor (3.3.1)", "aceite se grupo menor ≥ 20%", "(exigência só no gatilho A)"], LARANJA_P, LARANJA, LARANJA)
bloco(Q2[0], yb, Q2[1], hb, ["Resgate pela cor", "em 24 recortes, sem relva", "aceite se grupo menor ≥ 20%", "e arbitragem ≤ 25%", "senão, mantém a anterior"], LARANJA_P, LARANJA, LARANJA)

ym = ys + hs / 2
seta(C0[0] + C0[1], ym, C1[0], ym); seta(C1[0] + C1[1], ym, Q1[0], ym)
seta(Q1[0] + Q1[1], ym, Q2[0], ym); seta(Q2[0] + Q2[1], ym, F[0], ym)
texto(Q1[0] + Q1[1], ym - 0.3, Q2[0] - Q1[0] - Q1[1], 0.2, "sim", 9, INK, bold=True)
texto(Q2[0] + Q2[1], ym - 0.3, F[0] - Q2[0] - Q2[1], 0.2, "sim", 9, INK, bold=True)
# desvios: para baixo («não») e de volta à faixa de cima
xa = Q1[0] + 0.45; seta(xa, ys + hs, xa, yb); texto(xa + 0.08, ys + hs + 0.12, 0.6, 0.2, "não", 9, LARANJA, bold=True, alinh="l")
texto(xa + 0.08, ys + hs + 0.34, 1.8, 0.2, "treino: 9 de 57 vídeos", 8, INK2, alinh="l")
seta(Q1[0] + Q1[1], yb + 0.35, Q2[0] + 0.3, ys + hs)
xb = Q2[0] + 0.45; seta(xb, ys + hs, xb, yb); texto(xb + 0.08, ys + hs + 0.12, 0.6, 0.2, "não", 9, LARANJA, bold=True, alinh="l")
texto(xb + 0.08, ys + hs + 0.34, 1.8, 0.2, "11 de 159 vídeos", 8, INK2, alinh="l")
seta(Q2[0] + Q2[1], yb + 0.35, F[0] + 0.4, ys + hs)
texto(F[0] - 0.1, yb + 0.5, 1.9, 0.2, "resgate aceite em 8", 8, INK2)

prs.save(f"{AQ}/fig_auditoria.pptx"); print("-> fig_auditoria.pptx")
