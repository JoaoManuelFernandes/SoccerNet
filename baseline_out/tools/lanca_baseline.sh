#!/bin/bash
# Lança a baseline como unidade systemd --user própria, com limite de memória.
# Sobrevive ao VS Code fechar; NÃO sobrevive a logout/oomd matar a sessão (aí relançar).
cd /home/joao/soccernet
systemctl --user is-active baseline-test >/dev/null 2>&1 && { echo "já está a correr"; exit 0; }
systemd-run --user --unit=baseline-test --collect \
  -p MemoryMax=4500M -p MemoryHigh=4000M \
  -p WorkingDirectory=/home/joao/soccernet \
  /home/joao/soccernet/run_baseline_test.sh
