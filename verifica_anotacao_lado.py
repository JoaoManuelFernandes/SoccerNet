#!/usr/bin/env python3
"""Verifica a coerencia da anotacao de equipa (left/right) no SoccerNet-GSR.

Motivacao: Oo et al. (Broadcast2Pitch, WACV 2026) afirmam que os videos
SNGS-126, SNGS-131 e SNGS-197 estao "incorrectly annotated for the team
attribute" no conjunto de teste. Este script testa essa afirmacao.

Dois testes independentes, por video:

  Teste 1 (ancora do guarda-redes). O guarda-redes da equipa "left" tem de
  estar junto a baliza esquerda (x < 0) e o da equipa "right" junto a direita
  (x > 0). E o sinal mais fiavel de qual lado uma equipa ocupa, porque o
  guarda-redes nao acompanha as fases de ataque.

  Teste 2 (heuristica ingenua da media). Compara a posicao horizontal mediana
  dos jogadores de cada equipa e assinala os videos onde a ordem se inverte.
  E a suposicao usada por varias solucoes do desafio, incluindo a vencedora
  de 2025.

Uso:  python3 verifica_anotacao_lado.py [caminho_do_split]
"""
import json, statistics as st, os, sys, gc

BASE = sys.argv[1] if len(sys.argv) > 1 else '/home/joao/soccernet/data/SoccerNetGS/test'
MIN_DET_GK = 30   # deteccoes minimas para o guarda-redes contar

def main():
    vids = sorted(v for v in os.listdir(BASE) if v.startswith('SNGS-'))
    coerentes, incoerentes, sem_gk, invertidos = [], [], [], []

    for vid in vids:
        p = os.path.join(BASE, vid, 'Labels-GameState.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        gk, players = {}, {'left': [], 'right': []}
        for a in d['annotations']:
            at = a.get('attributes') or {}
            bp = a.get('bbox_pitch')
            if not bp:
                continue
            x = bp.get('x_bottom_middle')
            if x is None:
                continue
            role, team = at.get('role'), at.get('team')
            if role == 'goalkeeper' and team:
                gk.setdefault((a['track_id'], team), []).append(x)
            elif role == 'player' and team in players:
                players[team].append(x)
        del d
        gc.collect()

        vereditos = []
        for (tid, team), xs in gk.items():
            if len(xs) < MIN_DET_GK:
                continue
            mx = st.median(xs)
            ok = (mx < 0) if team == 'left' else (mx > 0)
            vereditos.append((tid, team, mx, ok))
        if not vereditos:
            sem_gk.append(vid)
        elif all(v[3] for v in vereditos):
            coerentes.append(vid)
        else:
            incoerentes.append((vid, vereditos))

        if players['left'] and players['right']:
            ml, mr = st.median(players['left']), st.median(players['right'])
            if ml >= mr:
                invertidos.append((vid, ml, mr))

    print(f"Conjunto: {BASE}")
    print(f"Videos analisados: {len(vids)}\n")
    print("TESTE 1 - ancora do guarda-redes")
    print(f"  coerentes             : {len(coerentes)}")
    print(f"  sem guarda-redes util : {len(sem_gk)}  {sem_gk}")
    print(f"  INCOERENTES           : {len(incoerentes)}")
    for vid, vs in incoerentes:
        for tid, team, mx, ok in vs:
            print(f"     {vid} track {tid}: team='{team}' x={mx:+.1f} -> {'ok' if ok else 'INVERTIDO'}")
    print(f"\nTESTE 2 - heuristica da media invertida: {len(invertidos)}")
    for vid, ml, mr in invertidos:
        print(f"     {vid}: left={ml:+.1f} m   right={mr:+.1f} m")

if __name__ == '__main__':
    main()
