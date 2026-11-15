#!/bin/bash
# Corre o código ORIGINAL do SoccerNet (worktree em /home/joao/baseline_soccernet)
# sobre os 49 vídeos de teste. Guarda estado + vídeo renderizado por vídeo.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH=/home/joao/baseline_soccernet/sn-gamestate
BL=/home/joao/baseline_soccernet/sn-gamestate
TL=/home/joao/soccernet/venv_tl/bin/tracklab
OUT=/home/joao/soccernet/baseline_out
mkdir -p "$OUT/states" "$OUT/videos" "$OUT/logs"
LOG="$OUT/progresso.log"
cd "$BL"

VIDS=$(ls -1 /home/joao/soccernet/data/SoccerNetGS/test | grep '^SNGS-' | sort)
N=0
for V in $VIDS; do
  N=$((N+1))
  [ -f "$OUT/videos/$V.mp4" ] && continue
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] PARADO: só ${FREE}G livres" >> "$LOG"; exit 1; fi
  echo "[$(date)] $V ($N/49) livre=${FREE}G" >> "$LOG"
  $TL -cn soccernet num_cores=3 dataset.eval_set=test dataset.nvid=-1 dataset.nframes=-1 \
      "dataset.vids_dict.test=[$V]" eval_tracking=False \
      visualization.cfg.save_videos=True \
      state.save_file="$OUT/states/$V.pklz" > "$OUT/logs/$V.log" 2>&1
  M=$(ls -t "$BL"/outputs/*/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$OUT/videos/$V.mp4"
    echo "  -> ok, vídeo $(du -h "$OUT/videos/$V.mp4"|cut -f1), estado $(du -h "$OUT/states/$V.pklz" 2>/dev/null|cut -f1)" >> "$LOG"
  else echo "  -> FALHOU (ver logs/$V.log)" >> "$LOG"; fi
  find "$BL/outputs" -mindepth 1 -maxdepth 2 -type d -mmin +5 -exec rm -rf {} + 2>/dev/null
done
echo "[$(date)] BASELINE TESTE CONCLUÍDA" >> "$LOG"
