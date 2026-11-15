#!/bin/bash
set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; unset PYTHONPATH
SN=/home/joao/soccernet/sn-gamestate; TL=/home/joao/soccernet/venv_tl/bin/tracklab
LOG=/home/joao/soccernet/team_exp/render_mudados.log
cd "$SN"
for E in "SNGS-039 valid $SN/docs/final_report/videos/valid" "SNGS-069 train $SN/docs/full_report/videos/train"; do
  set -- $E; V=$1; S=$2; DEST=$3
  echo "[$(date '+%F %T')] (2.ª passagem) $V ($S)" >> "$LOG"
  $TL -cn soccernet num_cores=2 dataset.eval_set=$S dataset.nvid=-1 dataset.nframes=-1 \
     "++dataset.vids_dict.$S=[$V]" eval_tracking=False "pipeline=[]" \
     state.load_file=/home/joao/soccernet/nosso_out/states/$V.pklz state.save_file=null \
     visualization.cfg.save_videos=True > "/home/joao/soccernet/team_exp/render_$V.log" 2>&1
  M=$(ls -t "$SN"/outputs/*/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv -f "$M" "$DEST/$V.mp4"; echo "  ok ($(du -h "$DEST/$V.mp4" | cut -f1))" >> "$LOG"; else echo "  FALHOU $V" >> "$LOG"; fi
  find "$SN/outputs" -mindepth 1 -maxdepth 2 -type d -mmin +2 -exec rm -rf {} + 2>/dev/null
done
echo "[$(date '+%F %T')] RENDER 2 CONCLUÍDO" >> "$LOG"
