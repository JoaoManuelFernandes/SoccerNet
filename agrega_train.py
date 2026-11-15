#!/usr/bin/env python3
"""Agrega as metricas do conjunto de treino: GS-HOTA por lote + JNR/equipa acumulados."""
import re, sys, glob, os
sys.path.insert(0, '/home/joao/soccernet')
from extract_gshota import extract

SN = '/home/joao/soccernet/sn-gamestate'

def tabela(md, seccao):
    """Le a 1a linha de dados da tabela que vem logo a seguir a '### <seccao>'
    dentro do bloco '## Global Summary'."""
    bloco = md.split('## Global Summary')[-1]
    m = re.search(rf'### {seccao}\b(.*?)(?=\n### |\Z)', bloco, re.S)
    if not m: return None
    for linha in m.group(1).splitlines():
        if linha.startswith('|') and '---' not in linha and 'Correct' not in linha:
            return [c.strip().replace('**','') for c in linha.strip('|').split('|')]
    return None

jnr_c = jnr_t = team_c = team_t = 0
lotes=[]
for log in sorted(glob.glob(f'{SN}/docs/train_logs/batch*.log')):
    n = os.path.basename(log)[5:-4]
    rep = f'{SN}/docs/train_reports/train-b{n}.md'
    if not os.path.exists(rep):        # lote a meio
        continue
    g = extract(log)
    md = open(rep).read()
    j, t = tabela(md, 'JNR'), tabela(md, r'Team / Team-side')
    ids = sorted(set(re.findall(r'SNGS-\d+', open(log, errors='ignore').read())))
    lotes.append((n, float(g) if g else None, j, t, ids))
    if j: jnr_c += int(j[0]); jnr_t += int(j[4])
    if t: team_c += int(t[0]); team_t += int(t[3])

print(f"| Lote | Vídeos | GS-HOTA | Número | Equipa |")
print(f"|---|---|---|---|---|")
for n,g,j,t,ids in lotes:
    vids = ", ".join(x.replace('SNGS-','') for x in ids)
    print(f"| {n} | {vids} | {g if g else '—'}% | {j[5] if j else '—'} | {t[4] if t else '—'} |")

gs = [g for _,g,_,_,_ in lotes if g]
print(f"\nLotes concluídos: {len(lotes)}  ({len(lotes)*3} vídeos de 57)")
print(f"GS-HOTA médio por lote : {sum(gs)/len(gs):.2f}%   (min {min(gs):.2f} / max {max(gs):.2f})")
print(f"Número  acumulado      : {jnr_c}/{jnr_t} = {100*jnr_c/jnr_t:.1f}%")
print(f"Equipa  acumulada      : {team_c}/{team_t} = {100*team_c/team_t:.1f}%")
