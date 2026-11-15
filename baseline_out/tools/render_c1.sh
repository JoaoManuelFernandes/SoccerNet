#!/bin/bash
# Renderiza o MP4 a partir do estado 100% original recalibrado (states_ocr_c1), sem correr módulos (CPU).
# Uso: render_ocr.sh <split> <SNGS-xxx>   -> baseline_out/videos/<V>.mp4 ; logs_render/<V>.log
set -u
SPLIT=$1; V=$2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH=/home/joao/baseline_soccernet/sn-gamestate
BL=/home/joao/baseline_soccernet/sn-gamestate
TL=/home/joao/soccernet/venv_tl/bin/tracklab
OUT=/home/joao/soccernet/baseline_out
mkdir -p "$OUT/logs_render" "$OUT/videos"
[ -s "$OUT/states_ocr_c1/$V.pklz" ] || { echo "$V: sem estado OCR"; exit 1; }
cd "$BL"
$TL -cn soccernet num_cores=3 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 \
  "dataset.vids_dict.$SPLIT=[$V]" eval_tracking=False "pipeline=[]" \
  state.load_file="$OUT/states_ocr_c1/$V.pklz" state.save_file=null \
  visualization.cfg.save_videos=True > "$OUT/logs_render/$V.log" 2>&1
rc=$?
M=$(ls -t "$BL"/outputs/*/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
if [ $rc -eq 0 ] && [ -n "$M" ]; then mv "$M" "$OUT/videos/$V.mp4"; echo "$V: render OCR ok ($(du -h "$OUT/videos/$V.mp4" | cut -f1))"
else echo "$V: render FALHOU rc=$rc"; fi
find "$BL/outputs" -mindepth 1 -maxdepth 2 -type d -mmin +5 -exec rm -rf {} + 2>/dev/null
exit $rc
