"""Figura 3.1 (Pipeline GSR, nove módulos) como PowerPoint editável, com a mesma geometria, cores e textos de
sn-gamestate/docs/figuras_tese/fig_pipeline_modulos.py. Cada módulo, cada seta, a legenda e cada fase são um grupo,
para se poder apagar ou mover com um clique (duplo clique entra no grupo para editar o texto).
Uso: PYTHONPATH=pptx_libs python3 gera_fig_pipeline_modulos_pptx.py -> fig_pipeline_modulos.pptx"""
import math, os, sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

AZUL, AZUL_ESC = "2A78D6", "1B56A3"
INK, INK2, GRID, BG, PALE, SOMBRA = "0B0B0B", "52514E", "D8D8D4", "FCFCFB", "E9F1FB", "D9DCDF"
BORDA, CIRC, SETA = "C9CCD1", "ECEEF1", "7A7D82"
mods = [("bbox_detector", "Deteção de pessoas", None), ("reid", "Reidentificação", "estendido"),
        ("track", "Tracking", None), ("pitch", "Pontos do campo", None), ("calibration", "Calibração", None),
        ("jersey_number_detect", "Número da camisola", "substituído"), ("tracklet_agg", "Agregação", None),
        ("team", "Equipa", "reescrito"), ("team_side", "Lado do campo", "reescrito")]
U = 13 * 0.775 / 132                     # escala da figura original (13 polegadas × 77,5% de eixos para 132 unidades); y invertido
LARG, ALT = 140, 60                       # 140 (e não 132) para caberem seis caixas na linha de cima
X = lambda x: Inches(x * U)
Y = lambda y: Inches((ALT - y) * U)
L = lambda v: Inches(v * U)

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(LARG * U), Inches(ALT * U)
sl = prs.slides.add_slide(prs.slide_layouts[6])
sl.background.fill.solid(); sl.background.fill.fore_color.rgb = RGBColor.from_string(BG)


def caixa(grupo, x, y, w, h, raio, fill, borda=None, lw=1.0, sombra=False, nome=None):
    """retângulo arredondado com canto inferior esquerdo em (x, y) nas unidades da figura"""
    s = grupo.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(x), Y(y + h), L(w), L(h))
    s.adjustments[0] = min(0.5, raio / min(w, h))
    s.fill.solid(); s.fill.fore_color.rgb = RGBColor.from_string(fill)
    if borda: s.line.color.rgb = RGBColor.from_string(borda); s.line.width = Pt(lw)
    else: s.line.fill.background()
    spPr = s._element.spPr
    ef = etree.SubElement(spPr, qn("a:effectLst"))          # sem a sombra do tema
    if sombra:                                              # sombra plana 0,5 para a direita e 0,6 para baixo, como no original
        dx, dy = 0.5 * U, 0.6 * U
        sh = etree.SubElement(ef, qn("a:outerShdw"), blurRad="0", dist=str(int(Inches(math.hypot(dx, dy)))),
                              dir=str(int(math.degrees(math.atan2(dy, dx)) * 60000)), algn="tl", rotWithShape="0")
        etree.SubElement(sh, qn("a:srgbClr"), val=SOMBRA)
    if nome: s.name = nome
    return s


def texto(grupo, cx, cy, txt, pt, cor, bold=False, mono=False, w=23.4, h=3.0, ancora="c", alinh="c", nome=None):
    """caixa de texto sem margens; (cx, cy) é o centro (ancora c), o fundo (b) ou o topo (t); alinh c ou l (então cx é a esquerda)"""
    x = cx - w / 2 if alinh == "c" else cx
    y0 = {"c": cy - h / 2, "b": cy, "t": cy - h}[ancora]
    tb = grupo.add_textbox(X(x), Y(y0 + h), L(w), L(h)); tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.word_wrap = True
    tf.vertical_anchor = {"c": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM, "t": MSO_ANCHOR.TOP}[ancora]
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if alinh == "c" else PP_ALIGN.LEFT
    r = p.add_run(); r.text = txt; f = r.font
    f.size = Pt(pt); f.bold = bold; f.name = "Courier New" if mono else "Arial"; f.color.rgb = RGBColor.from_string(cor)
    if nome: tb.name = nome
    return tb


def linha(grupo, x1, y1, x2, y2, seta=False, cor=SETA, lw=1.2):
    c = grupo.add_connector(MSO_CONNECTOR.STRAIGHT, X(x1), Y(y1), X(x2), Y(y2))
    c.line.color.rgb = RGBColor.from_string(cor); c.line.width = Pt(lw)
    if seta: etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle", w="med", len="med")
    return c


W, H, GAP = 20.6, 15.0, 2.5
rows = {0: 32.0, 1: 7.0}; x0 = {0: 1.5, 1: 1.5 + (W + GAP) / 2}; pos = []
for i, (key, nome, estado) in enumerate(mods):
    r = 0 if i < 6 else 1; c = i if r == 0 else i - 6   # 1-6 por imagem/deteção, 7-9 por vídeo
    x, y = x0[r] + c * (W + GAP), rows[r]; pos.append((x, y)); mudou = estado is not None
    g = sl.shapes.add_group_shape(); g.name = f"Módulo {i + 1} – {nome}"; gs = g.shapes
    caixa(gs, x, y, W, H, 1.6, PALE if mudou else "FFFFFF", AZUL if mudou else BORDA, 1.6 if mudou else 1.0, sombra=True, nome="caixa")
    if mudou: caixa(gs, x, y + H - 1.3, W, 1.3, 0.6, AZUL, nome="faixa azul")
    o = gs.add_shape(MSO_SHAPE.OVAL, X(x + 2.7 - 1.5), Y(y + H - 3.4 + 1.5), L(3.0), L(3.0)); o.name = "círculo"
    o.fill.solid(); o.fill.fore_color.rgb = RGBColor.from_string(AZUL if mudou else CIRC); o.line.fill.background()
    etree.SubElement(o._element.spPr, qn("a:effectLst"))
    tf = o.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; rr = p.add_run(); rr.text = str(i + 1)
    rr.font.size = Pt(7.4); rr.font.bold = True; rr.font.name = "Arial"; rr.font.color.rgb = RGBColor.from_string("FFFFFF" if mudou else INK2)
    texto(gs, x + W / 2, y + H - 7.0, nome, 8.6, INK if mudou else "2B2B2B", bold=True, nome="nome")
    texto(gs, x + W / 2, y + 4.2, key, 7.4, AZUL_ESC if mudou else INK2, mono=True, h=2.4, nome="módulo no código")
    if mudou: texto(gs, x + W / 2, y + 1.6, estado, 7.4, AZUL_ESC, bold=True, h=2.4, nome="estado")

for i in range(len(mods) - 1):
    (xa, ya), (xb, yb) = pos[i], pos[i + 1]
    g = sl.shapes.add_group_shape(); g.name = f"Seta {i + 1}→{i + 2}"; gs = g.shapes
    if i == 5:
        cx = xa + W / 2; cy = ya - 3.4; lx_ = xb - 3.2; my_ = yb + H / 2
        linha(gs, cx, ya, cx, cy); linha(gs, cx, cy, lx_, cy); linha(gs, lx_, cy, lx_, my_); linha(gs, lx_, my_, xb, my_, seta=True)
    else:
        linha(gs, xa + W, ya + H / 2, xb, yb + H / 2, seta=True)

# faixas por nível do TrackLab: o módulo 6 (número) é de nível deteção, por isso fica na linha de cima
for r, rot, n in ((0, "POR IMAGEM E POR DETEÇÃO", 6), (1, "POR VÍDEO, SOBRE TRACKLETS COMPLETOS", 3)):
    g = sl.shapes.add_group_shape(); g.name = f"Fase – {rot.lower()}"; gs = g.shapes
    texto(gs, x0[r], rows[r] + H + 2.2, rot, 7.6, INK2, bold=True, w=60, h=2.4, ancora="b", alinh="l")
    linha(gs, x0[r], rows[r] + H + 1.8, x0[r] + (n - 1) * (W + GAP) + W, rows[r] + H + 1.8, cor=GRID, lw=1.0)

texto(sl.shapes, 1.5, 58.5, "Pipeline GSR (módulos)", 13, INK, bold=True, w=60, h=4.0, ancora="t", alinh="l", nome="Título")
if "--com-legenda" in sys.argv:                  # o João tirou a legenda no JF-6 (a legenda da figura já explica o azul)
    g = sl.shapes.add_group_shape(); g.name = "Legenda"; gs = g.shapes
    lx, ly = 98.5, 55.5
    caixa(gs, lx, ly - 1.2, 3.2, 2.4, 0.5, PALE, AZUL, 1.4)
    texto(gs, lx + 4.2, ly, "módulo tocado neste trabalho", 8, INK2, w=28, h=2.4, alinh="l")
    caixa(gs, lx, ly - 4.8, 3.2, 2.4, 0.5, "FFFFFF", BORDA, 1.0)
    texto(gs, lx + 4.2, ly - 3.6, "código e pesos originais", 8, INK2, w=28, h=2.4, alinh="l")

saida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_pipeline_modulos.pptx")
prs.save(saida); print("->", saida)
