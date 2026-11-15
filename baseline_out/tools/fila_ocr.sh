#!/bin/bash
# Espera a baseline acabar e corre a passagem OCR original em todos os estados do split.
# Uso: fila_ocr.sh <split>
SPLIT=${1:-test}
while systemctl --user is-active baseline-test >/dev/null 2>&1; do sleep 120; done
OUT=/home/joao/soccernet/baseline_out
for f in "$OUT"/states/SNGS-*.pklz; do
  V=$(basename "$f" .pklz)
  [ -d "/home/joao/soccernet/data/SoccerNetGS/$SPLIT/$V" ] || continue
  echo "[$(date '+%F %T')] OCR $V" >> "$OUT/progresso_ocr.log"
  "$OUT/tools/passagem_ocr.sh" "$SPLIT" "$V" >> "$OUT/progresso_ocr.log" 2>&1
done
echo "[$(date '+%F %T')] fila OCR $SPLIT terminada" >> "$OUT/progresso_ocr.log"
