#!/usr/bin/env python3
"""Extrai o GS-HOTA de um log do tracklab.

O console 'rich' parte a linha do resultado em varias linhas e insere codigos
ANSI, por isso um grep simples nao apanha o valor. Aqui limpa-se o ANSI, junta-se
o texto todo e procura-se o padrao no fluxo continuo.
"""
import re, sys

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PAT  = re.compile(r"performance\s+GS-HOTA\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%")

def extract(path):
    with open(path, errors="ignore") as fh:
        txt = ANSI.sub("", fh.read())
    # o rich alinha a continuacao com espacos; colapsar tudo num fluxo unico
    flat = re.sub(r"\s+", " ", txt)
    vals = PAT.findall(flat)
    return vals[-1] if vals else None

if __name__ == "__main__":
    v = extract(sys.argv[1])
    print(v if v else "FALHOU")
