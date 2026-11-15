#!/bin/bash
# Relança a cadeia completa como unidade systemd --user com tecto de memória.
systemctl --user is-active baseline-tudo >/dev/null 2>&1 && { echo "já está a correr"; exit 0; }
systemd-run --user --unit=baseline-tudo --collect -p MemoryMax=4500M -p MemoryHigh=4000M \
  -p WorkingDirectory=/home/joao/soccernet /home/joao/soccernet/baseline_out/tools/corre_tudo.sh
