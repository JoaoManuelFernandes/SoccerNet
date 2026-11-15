#!/bin/bash
# Reescreve a tabela de comparação de 30 em 30 min (leve, CPU, venv_metrics).
while true; do
  /home/joao/soccernet/venv_metrics/bin/python /home/joao/soccernet/baseline_out/tools/compara_por_video.py > /home/joao/soccernet/baseline_out/tools/compara_loop.log 2>&1
  sleep 1800
done
