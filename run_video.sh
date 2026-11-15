#!/bin/bash
# Corre o pipeline SoccerNet-GSR num video externo (.mp4) e grava o minimap.
#
# Uso:  bash run_video.sh [NFRAMES]
#   teste rapido : bash run_video.sh 40
#   clip completo: bash run_video.sh 1069
#   IMPORTANTE: NAO usar -1. nframes=-1 REATIVA o GS-HOTA, que crasha sem
#   ground truth. Usa sempre um numero (1069 = clip completo).
#
# Requisitos de RAM: fecha browser/Overleaf antes (a maquina so tem 7.6GB).
# Saida: outputs/sn-gamestate/<data>/<hora>/visualization/videos/0.mp4
set -e
source /home/joao/miniconda3/etc/profile.d/conda.sh
conda activate tracklab
cd /home/joao/soccernet/sn-gamestate

# VIDEO pode ser um .mp4 local OU um URL do YouTube (usa yt-dlp).
# Ex.: VIDEO="https://www.youtube.com/watch?v=XXXX" bash run_video.sh 300
VIDEO="${VIDEO:-/home/joao/soccernet/video_test.mp4}"
NFRAMES="${1:-1069}"

echo "RAM antes:"; free -h | head -2
echo "A correr $NFRAMES frames de: $VIDEO"

tracklab -cn soccernet \
  dataset=youtube \
  "dataset.video_path=$VIDEO" \
  dataset.nvid=1 \
  dataset.eval_set=val \
  dataset.nframes="$NFRAMES" \
  visualization.cfg.save_videos=True \
  visualization.cfg.pitch.pitch_scale=3 \
  2>&1 | tee /tmp/video_run.log

echo ""
echo "Minimap em: $(find outputs/sn-gamestate -name '0.mp4' -newermt '-30 minutes' 2>/dev/null | tail -1)"
