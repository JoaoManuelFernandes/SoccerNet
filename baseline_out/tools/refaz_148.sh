#!/bin/bash
# Refaz só o SNGS-148 para a figura do lado trocado (Cap. 4): a baseline original com a mesma cadeia dos
# 160 vídeos (pipeline -> OCR original -> recalibração frame a frame) e os dois MP4, baseline e nosso.
# O estado do nosso sistema já existe (nosso_out/states); só se renderiza.
set -u
V=SNGS-148; SPLIT=test
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
BL=/home/joao/baseline_soccernet/sn-gamestate; SN=/home/joao/soccernet/sn-gamestate
TL=/home/joao/soccernet/venv_tl/bin/tracklab; PYM=/home/joao/soccernet/venv_metrics/bin/python
OUT=/home/joao/soccernet/baseline_out; LOG="$OUT/refaz_148.log"
mkdir -p "$OUT/states" "$OUT/logs" "$OUT/states_ocr_norm" "$OUT/states_ocr_c1" "$OUT/logs_c1" "$OUT/logs_render" "$OUT/videos"
msg() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
msg "início"

# 1) pipeline original (como em corre_tudo.sh, mas sem MP4: o vídeo final sai do estado recalibrado)
if [ ! -s "$OUT/states/$V.pklz" ]; then
  (cd "$BL" && PYTHONPATH="$BL" $TL -cn soccernet num_cores=3 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 \
      "dataset.vids_dict.$SPLIT=[$V]" eval_tracking=False visualization.cfg.save_videos=False \
      state.save_file="$OUT/states/$V.pklz" > "$OUT/logs/$V.log" 2>&1)
  [ -s "$OUT/states/$V.pklz" ] || { msg "1) baseline FALHOU (logs/$V.log)"; exit 1; }
fi
msg "1) baseline ok"

# 2) OCR original (MMOCR)
PYTHONPATH="$BL" "$OUT/tools/passagem_ocr.sh" $SPLIT $V >> "$LOG" 2>&1 || { msg "2) OCR FALHOU"; exit 1; }

# 3) recalibração frame a frame (como em recalibra_c1.sh)
if [ ! -s "$OUT/states_ocr_c1/$V.pklz" ]; then
  $PYM "$OUT/tools/normaliza_kp.py" "$OUT/states_ocr/$V.pklz" "$OUT/states_ocr_norm/$V.pklz" >> "$LOG" 2>&1 || { msg "3) normalização FALHOU"; exit 1; }
  (cd "$BL" && PYTHONPATH="$BL" $TL -cn soccernet num_cores=2 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 \
      "++dataset.vids_dict.$SPLIT=[$V]" "pipeline=[calibration,team_side]" modules.calibration.batch_size=1 \
      eval_tracking=False visualization.cfg.save_videos=False \
      state.load_file="$OUT/states_ocr_norm/$V.pklz" state.save_file="$OUT/states_ocr_c1/$V.pklz" > "$OUT/logs_c1/$V.log" 2>&1)
  [ -s "$OUT/states_ocr_c1/$V.pklz" ] || { msg "3) recalibração FALHOU (logs_c1/$V.log)"; exit 1; }
fi
msg "3) recalibração ok"

# 4) MP4 da baseline recalibrada
PYTHONPATH="$BL" "$OUT/tools/render_c1.sh" $SPLIT $V >> "$LOG" 2>&1 || { msg "4) render baseline FALHOU"; exit 1; }
mv "$OUT/videos/$V.mp4" "$OUT/videos/${V}_baseline.mp4"

# 5) MP4 do nosso sistema, a partir do estado final
(cd "$SN" && $TL -cn soccernet num_cores=3 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 \
    "dataset.vids_dict.$SPLIT=[$V]" eval_tracking=False "pipeline=[]" \
    state.load_file=/home/joao/soccernet/nosso_out/states/$V.pklz state.save_file=null \
    visualization.cfg.save_videos=True > "$OUT/logs_render/$V.nosso.log" 2>&1)
M=$(ls -t "$SN"/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
[ -n "$M" ] && mv "$M" "$OUT/videos/${V}_nosso.mp4" || { msg "5) render nosso FALHOU"; exit 1; }
msg "5) videos/${V}_baseline.mp4 e videos/${V}_nosso.mp4"

# 6) exemplo da equipa de amarelo dada como arbitragem (falha igual nos dois sistemas): só o nosso render
A=SNGS-050
(cd "$SN" && $TL -cn soccernet num_cores=3 dataset.eval_set=valid dataset.nvid=-1 dataset.nframes=-1 \
    "dataset.vids_dict.valid=[$A]" eval_tracking=False "pipeline=[]" \
    state.load_file=/home/joao/soccernet/nosso_out/states/$A.pklz state.save_file=null \
    visualization.cfg.save_videos=True > "$OUT/logs_render/$A.nosso.log" 2>&1)
M=$(ls -t "$SN"/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
[ -n "$M" ] && mv "$M" "$OUT/videos/${A}_nosso.mp4" || { msg "6) render $A FALHOU"; exit 1; }
msg "FIM: videos/${V}_baseline.mp4, videos/${V}_nosso.mp4 e videos/${A}_nosso.mp4"
