#!/bin/bash
# Segunda passagem, OCR original (MMOCR DBNet+SAR) a partir de um estado da baseline.
# Uso: passagem_ocr.sh <split> <SNGS-xxx>   -> baseline_out/states_ocr/<V>.pklz  + logs_ocr/<V>.log
set -u
SPLIT=$1; V=$2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/joao/baseline_soccernet/sn-gamestate
BL=/home/joao/baseline_soccernet/sn-gamestate
PY=/home/joao/soccernet/venv_bl/bin/python
TL=/home/joao/soccernet/venv_tl/bin/tracklab
OUT=/home/joao/soccernet/baseline_out
mkdir -p "$OUT/states_ocr" "$OUT/logs_ocr"
[ -s "$OUT/states_ocr/$V.pklz" ] && { echo "$V: já tem estado OCR"; exit 0; }
[ -s "$OUT/states/$V.pklz" ] || { echo "$V: sem estado da baseline"; exit 1; }
cd "$BL"
$PY $TL -cn soccernet num_cores=3 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 \
  "dataset.vids_dict.$SPLIT=[$V]" eval_tracking=False visualization.cfg.save_videos=False \
  modules/jersey_number_detect=mmocr modules.jersey_number_detect.batch_size=4 "pipeline=[jersey_number_detect,tracklet_agg]" \
  state.load_file="$OUT/states/$V.pklz" state.save_file="$OUT/states_ocr/$V.pklz" \
  > "$OUT/logs_ocr/$V.log" 2>&1
rc=$?
[ $rc -eq 0 ] && [ -s "$OUT/states_ocr/$V.pklz" ] && echo "$V: OCR ok ($(du -h "$OUT/states_ocr/$V.pklz" | cut -f1))" || { echo "$V: OCR FALHOU rc=$rc"; rm -f "$OUT/states_ocr/$V.pklz"; }
find "$BL/outputs" -mindepth 1 -maxdepth 2 -type d -mmin +5 -exec rm -rf {} + 2>/dev/null
exit $rc
