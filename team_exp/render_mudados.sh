#!/bin/bash
# Re-renderiza os MP4 dos vídeos cuja equipa mudou, a partir dos estados já corrigidos (sem correr módulos).
set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; unset PYTHONPATH
SN=/home/joao/soccernet/sn-gamestate; TL=/home/joao/soccernet/venv_tl/bin/tracklab
LOG=/home/joao/soccernet/team_exp/render_mudados.log
declare -A SPLIT=( [SNGS-025]=valid [SNGS-027]=valid [SNGS-028]=valid [SNGS-030]=valid [SNGS-036]=valid \
                   [SNGS-038]=valid [SNGS-039]=valid [SNGS-091]=valid [SNGS-069]=train [SNGS-131]=test )
cd "$SN"
VIDS=${@:-${!SPLIT[@]}}
for V in $VIDS; do
  S=${SPLIT[$V]}
  case $S in valid|test) DEST="$SN/docs/final_report/videos/$S";; train) DEST="$SN/docs/full_report/videos/train";; esac
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9'); [ "$FREE" -ge 4 ] || { echo "[$(date '+%F %T')] DISCO < 4 GB, parado" >> "$LOG"; exit 1; }
  echo "[$(date '+%F %T')] $V ($S) -> $DEST" >> "$LOG"
  $TL -cn soccernet num_cores=2 dataset.eval_set=$S dataset.nvid=-1 dataset.nframes=-1 \
     "++dataset.vids_dict.$S=[$V]" eval_tracking=False "pipeline=[]" \
     state.load_file=/home/joao/soccernet/nosso_out/states/$V.pklz state.save_file=null \
     visualization.cfg.save_videos=True > "/home/joao/soccernet/team_exp/render_$V.log" 2>&1
  M=$(ls -t "$SN"/outputs/*/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then
    [ -f "$DEST/$V.mp4" ] && mv "$DEST/$V.mp4" "$DEST/$V.mp4.antigo"
    mv "$M" "$DEST/$V.mp4"; rm -f "$DEST/$V.mp4.antigo"
    echo "  ok ($(du -h "$DEST/$V.mp4" | cut -f1))" >> "$LOG"
  else echo "  FALHOU $V" >> "$LOG"; fi
  find "$SN/outputs" -mindepth 1 -maxdepth 2 -type d -mmin +2 -exec rm -rf {} + 2>/dev/null
done
echo "[$(date '+%F %T')] RENDER CONCLUÍDO ($(grep -c '  ok' "$LOG"))" >> "$LOG"
