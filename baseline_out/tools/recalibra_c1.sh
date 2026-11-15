#!/bin/bash
# Corrige a calibração da baseline original: a configuração por omissão (calibration.batch_size=8) aplica a
# homografia do 1.º frame de cada lote aos outros 7 (NBJW_Calib usa metadatas["keypoints"][0]). Reprocessa,
# a partir dos estados 100% originais (states_ocr, com keypoints), só [calibration, team_side] com batch_size=1,
# módulos originais, e grava em states_ocr_c1. O número (MMOCR) e o agrupamento ficam como estavam.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH=/home/joao/baseline_soccernet/sn-gamestate
BL=/home/joao/baseline_soccernet/sn-gamestate; TL=/home/joao/soccernet/venv_tl/bin/tracklab; PYM=/home/joao/soccernet/venv_metrics/bin/python
OUT=/home/joao/soccernet/baseline_out; LOG="$OUT/progresso_c1.log"
mkdir -p "$OUT/states_ocr_norm" "$OUT/states_ocr_c1" "$OUT/logs_c1"
cd "$BL"
for SPLIT in test valid train; do
  for F in $(ls -1 "$OUT/states_ocr"/SNGS-*.pklz | sort); do
    V=$(basename "$F" .pklz); [ -d /home/joao/soccernet/data/SoccerNetGS/$SPLIT/$V ] || continue
    [ -s "$OUT/states_ocr_c1/$V.pklz" ] && continue
    FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9'); [ "$FREE" -ge 3 ] || { echo "[$(date '+%F %T')] DISCO < 3 GB, parado" >> "$LOG"; exit 1; }
    echo "[$(date '+%F %T')] $SPLIT $V recalibração batch=1" >> "$LOG"
    $PYM "$OUT/tools/normaliza_kp.py" "$F" "$OUT/states_ocr_norm/$V.pklz" >> "$LOG" 2>&1 || { echo "  FALHOU normalização $V" >> "$LOG"; continue; }
    F="$OUT/states_ocr_norm/$V.pklz"
    $TL -cn soccernet num_cores=2 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 "++dataset.vids_dict.$SPLIT=[$V]" \
       "pipeline=[calibration,team_side]" modules.calibration.batch_size=1 eval_tracking=False visualization.cfg.save_videos=False \
       state.load_file="$F" state.save_file="$OUT/states_ocr_c1/$V.pklz" > "$OUT/logs_c1/$V.log" 2>&1
    if [ -s "$OUT/states_ocr_c1/$V.pklz" ]; then
      [ "$(stat -c %s "$OUT/states_ocr_c1/$V.pklz")" -gt 30000000 ] && $PYM "$OUT/tools/slim_estado.py" "$OUT/states_ocr_c1/$V.pklz" "$OUT/states_ocr_c1/$V.pklz" >> "$LOG" 2>&1
      echo "  ok ($(du -h "$OUT/states_ocr_c1/$V.pklz" | cut -f1))" >> "$LOG"; rm -f "$OUT/states_ocr_norm/$V.pklz"
    else echo "  FALHOU $V" >> "$LOG"; rm -f "$OUT/states_ocr_c1/$V.pklz"; fi
    find "$BL/outputs" -mindepth 2 -maxdepth 2 -type d -mmin +2 -exec rm -rf {} + 2>/dev/null
  done
done
echo "[$(date '+%F %T')] RECALIBRAÇÃO CONCLUÍDA ($(ls "$OUT/states_ocr_c1" | wc -l))" >> "$LOG"
