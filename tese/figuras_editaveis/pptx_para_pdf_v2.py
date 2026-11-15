"""Desenha o primeiro diapositivo de um .pptx simples (retângulos, retângulos arredondados, elipses, linhas com seta,
caixas de texto, grupos, imagens; transparência no preenchimento e nas imagens desde 12/10) com o matplotlib e grava PDF e PNG cortados ao conteúdo, com as letras das outras figuras
da tese (Arial -> DejaVu Sans, Courier New -> DejaVu Sans Mono). Serve para as figuras editáveis desta pasta.
Uso: PYTHONPATH=pptx_libs python3 pptx_para_pdf.py fig.pptx [saida_sem_extensao]"""
import sys, os, math, io
import matplotlib.image as mpimg
from matplotlib.colors import to_rgba
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle, FancyArrowPatch
from pptx import Presentation
from pptx.oxml.ns import qn

EMU = 914400.0
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0]
prs = Presentation(src); SW, SH = prs.slide_width / EMU, prs.slide_height / EMU; sl = prs.slides[0]
fig = plt.figure(figsize=(SW, SH)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, SW); ax.set_ylim(SH, 0); ax.axis("off"); ax.set_autoscale_on(False)
bg = sl._element.find(".//" + qn("p:bg") + "//" + qn("a:srgbClr"))
fig.patch.set_facecolor("#" + bg.get("val") if bg is not None else "white")
z = [0]


def cor(el, caminho):
    c = el.find(caminho)
    if c is None: return None
    s = c.find(qn("a:srgbClr"))
    return "#" + s.get("val") if s is not None else None


def alfa(el, caminho):
    """transparência de um a:srgbClr (a:alpha, 100000 = opaco)"""
    c = el.find(caminho) if el is not None else None
    a = c.find(qn("a:srgbClr") + "/" + qn("a:alpha")) if c is not None else None
    return int(a.get("val")) / 100000 if a is not None else 1.0


def geo(sp, T):
    """posição e tamanho do shape em polegadas no diapositivo, aplicando as transformações dos grupos"""
    xf = sp.find(".//" + qn("a:xfrm"))
    off, ext = xf.find(qn("a:off")), xf.find(qn("a:ext"))
    x, y, w, h = int(off.get("x")), int(off.get("y")), int(ext.get("cx")), int(ext.get("cy"))
    X0, Y0, sx, sy = T
    return (X0 + x * sx) / EMU, (Y0 + y * sy) / EMU, w * sx / EMU, h * sy / EMU, xf.get("flipH") == "1", xf.get("flipV") == "1"


def desenha_texto(sp, x, y, w, h):
    txb = sp.find(qn("p:txBody"))
    if txb is None: return
    bp = txb.find(qn("a:bodyPr")); anc = bp.get("anchor", "t") if bp is not None else "t"
    mg = lambda k, d: int(bp.get(k, d)) / EMU if bp is not None else d / EMU
    l, r, t, b = mg("lIns", 91440), mg("rIns", 91440), mg("tIns", 45720), mg("bIns", 45720)
    linhas = []
    for p in txb.findall(qn("a:p")):
        ppr = p.find(qn("a:pPr")); al = ppr.get("algn", "l") if ppr is not None else "l"
        for run in p.findall(qn("a:r")):
            rp = run.find(qn("a:rPr")); t_ = run.find(qn("a:t")).text or ""
            if not t_.strip(): continue
            sz = int(rp.get("sz", 1800)) / 100; bold = rp.get("b") == "1"
            lat = rp.find(qn("a:latin")); fam = lat.get("typeface") if lat is not None else "Arial"
            linhas.append((t_, sz, bold, "DejaVu Sans Mono" if "Courier" in fam or "Mono" in fam else "DejaVu Sans",
                           cor(rp, qn("a:solidFill")) or "#000000", al))
    if not linhas: return
    t_, sz, bold, fam, c, al = linhas[0]
    tx = {"l": x + l, "ctr": x + w / 2, "r": x + w - r}.get(al, x + l)
    ty = {"t": y + t, "ctr": y + h / 2, "b": y + h - b}.get(anc, y + t)
    ax.text(tx, ty, " ".join(li[0] for li in linhas), fontsize=sz, fontweight="bold" if bold else "normal", family=fam, color=c,
            ha={"l": "left", "ctr": "center", "r": "right"}.get(al, "left"), va={"t": "top", "ctr": "center", "b": "bottom"}.get(anc, "top"),
            zorder=z[0] + 0.5)


def desenha(el, T):
    for sp in el:
        tag = sp.tag.split("}")[1]
        if tag == "grpSp":
            xf = sp.find(qn("p:grpSpPr")).find(qn("a:xfrm"))
            off, ext, coff, cext = (xf.find(qn(k)) for k in ("a:off", "a:ext", "a:chOff", "a:chExt"))
            gx, gy = int(off.get("x")), int(off.get("y")); X0, Y0, sx, sy = T
            ex, ey = int(ext.get("cx")), int(ext.get("cy")); cx_, cy_ = int(cext.get("cx")) or 1, int(cext.get("cy")) or 1
            nsx, nsy = ex / cx_, ey / cy_
            desenha(sp, (X0 + (gx - int(coff.get("x")) * nsx) * sx, Y0 + (gy - int(coff.get("y")) * nsy) * sy, sx * nsx, sy * nsy))
            continue
        if tag == "pic":                                             # imagem embutida, com transparência opcional (a:alphaModFix)
            z[0] += 1; x, y, w, h, fh, fv = geo(sp, T); blip = sp.find(".//" + qn("a:blip"))
            img = mpimg.imread(io.BytesIO(sl.part.related_part(blip.get(qn("r:embed"))).blob), format="png")
            am = blip.find(qn("a:alphaModFix")); a_ = int(am.get("amt")) / 100000 if am is not None else 1.0
            ax.imshow(img, extent=(x, x + w, y + h, y), aspect="auto", alpha=a_, zorder=z[0], interpolation="antialiased")
            continue
        if tag not in ("sp", "cxnSp"): continue
        z[0] += 1; x, y, w, h, fh, fv = geo(sp, T)
        spPr = sp.find(qn("p:spPr")); pg = spPr.find(qn("a:prstGeom")); prst = pg.get("prst") if pg is not None else "rect"
        fill = cor(spPr, qn("a:solidFill")); ln = spPr.find(qn("a:ln"))
        lc = cor(ln, qn("a:solidFill")) if ln is not None and ln.find(qn("a:noFill")) is None else None
        lw = int(ln.get("w", 12700)) / 12700 if ln is not None else 1.0
        if tag == "cxnSp" or prst in ("line", "straightConnector1"):
            x1, x2 = (x + w, x) if fh else (x, x + w); y1, y2 = (y + h, y) if fv else (y, y + h)
            seta = ln is not None and ln.find(qn("a:tailEnd")) is not None and ln.find(qn("a:tailEnd")).get("type", "none") != "none"
            if seta: ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=lc or "black", lw=lw,
                                                  shrinkA=0, shrinkB=0, zorder=z[0]))
            else: ax.plot([x1, x2], [y1, y2], color=lc or "black", lw=lw, zorder=z[0], solid_capstyle="butt")
            continue
        sh = spPr.find(".//" + qn("a:outerShdw"))
        if sh is not None and (fill or lc):
            d, a = int(sh.get("dist", 0)) / EMU, int(sh.get("dir", 0)) / 60000
            dx, dy = d * math.cos(math.radians(a)), d * math.sin(math.radians(a))
        kw = dict(fc=to_rgba(fill, alfa(spPr, qn("a:solidFill"))) if fill else "none", ec=lc or "none", lw=lw if lc else 0, zorder=z[0])
        if prst == "roundRect":
            av = spPr.find(".//" + qn("a:gd")); adj = int(av.get("fmla").split()[1]) / 100000 if av is not None else 0.16667
            rad = adj * min(w, h); mk = lambda xx, yy, **k: FancyBboxPatch((xx, yy), w, h, boxstyle=f"round,pad=0,rounding_size={rad}", **k)
        elif prst == "ellipse": mk = lambda xx, yy, **k: Ellipse((xx + w / 2, yy + h / 2), w, h, **k)
        else: mk = lambda xx, yy, **k: Rectangle((xx, yy), w, h, **k)
        if sh is not None and (fill or lc): ax.add_patch(mk(x + dx, y + dy, fc=cor(sh, ".") or "#d9dcdf", ec="none", zorder=z[0] - 0.1))
        if fill or lc: ax.add_patch(mk(x, y, **kw))
        desenha_texto(sp, x, y, w, h)


desenha(sl.shapes._spTree, (0, 0, 1.0, 1.0))
for ext in ("pdf", "png"): fig.savefig(f"{out}.{ext}", bbox_inches="tight", pad_inches=0.04, dpi=220, facecolor=fig.get_facecolor())
print("->", out + ".pdf", "e .png")
