#!/bin/bash
# Vigia de memória para uma unidade systemd --user: pára-a se a RAM disponível cair abaixo de 500 MB
# ou se a pressão de memória (some avg10) passar 25%, antes de o systemd-oomd matar a sessão (50% durante 20 s).
# Uso: vigia_memoria.sh <unidade>
U=$1; LOG=/home/joao/soccernet/baseline_out/vigia_$U.log
sleep 5
while systemctl --user is-active "$U" >/dev/null 2>&1; do
  AV=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  PR=$(awk '/^some/ {split($2,a,"="); print int(a[2])}' /proc/pressure/memory)
  if [ "$AV" -lt 500 ] || [ "$PR" -gt 25 ]; then
    echo "[$(date '+%F %T')] PAREI $U: disponível ${AV} MB, pressão ${PR}%" >> "$LOG"
    systemctl --user stop "$U"; exit 1
  fi
  sleep 3
done
echo "[$(date '+%F %T')] $U terminou" >> "$LOG"
