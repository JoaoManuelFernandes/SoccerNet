#!/bin/bash
# Corre o dataset de TREINO (57 videos) em lotes de 3, com GS-HOTA.
# NAO usa set -e: se um lote falhar, continua para o seguinte.
exec > >(tee -a /tmp/train_batches.log) 2>&1
source /home/joao/miniconda3/etc/profile.d/conda.sh
conda activate tracklab
cd /home/joao/soccernet/sn-gamestate

echo "[$(date)] A aguardar que nao haja tracklab a correr..."
for i in $(seq 1 720); do pgrep -f "tracklab -cn soccernet" >/dev/null || break; sleep 5; done
echo "[$(date)] Arranque dos lotes de treino."

echo "===== LOTE 1/19 : SNGS-060 SNGS-061 SNGS-062 ====="
echo "[$(date)] lote 1 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-060,SNGS-061,SNGS-062]' \
  state.save_file=states/train-batch01.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 1 fim"

echo "===== LOTE 2/19 : SNGS-063 SNGS-064 SNGS-065 ====="
echo "[$(date)] lote 2 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-063,SNGS-064,SNGS-065]' \
  state.save_file=states/train-batch02.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 2 fim"

echo "===== LOTE 3/19 : SNGS-066 SNGS-067 SNGS-068 ====="
echo "[$(date)] lote 3 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-066,SNGS-067,SNGS-068]' \
  state.save_file=states/train-batch03.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 3 fim"

echo "===== LOTE 4/19 : SNGS-069 SNGS-070 SNGS-071 ====="
echo "[$(date)] lote 4 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-069,SNGS-070,SNGS-071]' \
  state.save_file=states/train-batch04.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 4 fim"

echo "===== LOTE 5/19 : SNGS-072 SNGS-073 SNGS-074 ====="
echo "[$(date)] lote 5 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-072,SNGS-073,SNGS-074]' \
  state.save_file=states/train-batch05.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 5 fim"

echo "===== LOTE 6/19 : SNGS-075 SNGS-076 SNGS-077 ====="
echo "[$(date)] lote 6 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-075,SNGS-076,SNGS-077]' \
  state.save_file=states/train-batch06.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 6 fim"

echo "===== LOTE 7/19 : SNGS-097 SNGS-098 SNGS-099 ====="
echo "[$(date)] lote 7 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-097,SNGS-098,SNGS-099]' \
  state.save_file=states/train-batch07.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 7 fim"

echo "===== LOTE 8/19 : SNGS-100 SNGS-101 SNGS-102 ====="
echo "[$(date)] lote 8 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-100,SNGS-101,SNGS-102]' \
  state.save_file=states/train-batch08.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 8 fim"

echo "===== LOTE 9/19 : SNGS-103 SNGS-104 SNGS-105 ====="
echo "[$(date)] lote 9 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-103,SNGS-104,SNGS-105]' \
  state.save_file=states/train-batch09.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 9 fim"

echo "===== LOTE 10/19 : SNGS-106 SNGS-107 SNGS-108 ====="
echo "[$(date)] lote 10 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-106,SNGS-107,SNGS-108]' \
  state.save_file=states/train-batch10.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 10 fim"

echo "===== LOTE 11/19 : SNGS-109 SNGS-110 SNGS-111 ====="
echo "[$(date)] lote 11 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-109,SNGS-110,SNGS-111]' \
  state.save_file=states/train-batch11.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 11 fim"

echo "===== LOTE 12/19 : SNGS-112 SNGS-113 SNGS-114 ====="
echo "[$(date)] lote 12 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-112,SNGS-113,SNGS-114]' \
  state.save_file=states/train-batch12.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 12 fim"

echo "===== LOTE 13/19 : SNGS-115 SNGS-151 SNGS-152 ====="
echo "[$(date)] lote 13 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-115,SNGS-151,SNGS-152]' \
  state.save_file=states/train-batch13.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 13 fim"

echo "===== LOTE 14/19 : SNGS-153 SNGS-154 SNGS-155 ====="
echo "[$(date)] lote 14 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-153,SNGS-154,SNGS-155]' \
  state.save_file=states/train-batch14.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 14 fim"

echo "===== LOTE 15/19 : SNGS-156 SNGS-157 SNGS-158 ====="
echo "[$(date)] lote 15 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-156,SNGS-157,SNGS-158]' \
  state.save_file=states/train-batch15.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 15 fim"

echo "===== LOTE 16/19 : SNGS-159 SNGS-160 SNGS-161 ====="
echo "[$(date)] lote 16 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-159,SNGS-160,SNGS-161]' \
  state.save_file=states/train-batch16.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 16 fim"

echo "===== LOTE 17/19 : SNGS-162 SNGS-163 SNGS-164 ====="
echo "[$(date)] lote 17 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-162,SNGS-163,SNGS-164]' \
  state.save_file=states/train-batch17.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 17 fim"

echo "===== LOTE 18/19 : SNGS-165 SNGS-166 SNGS-167 ====="
echo "[$(date)] lote 18 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-165,SNGS-166,SNGS-167]' \
  state.save_file=states/train-batch18.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 18 fim"

echo "===== LOTE 19/19 : SNGS-168 SNGS-169 SNGS-170 ====="
echo "[$(date)] lote 19 inicio"
free -h | head -2
tracklab -cn soccernet \
  dataset.eval_set=train \
  dataset.nvid=-1 \
  dataset.nframes=-1 \
  'dataset.vids_dict.train=[SNGS-168,SNGS-169,SNGS-170]' \
  state.save_file=states/train-batch19.pklz \
  2>&1 | tail -60
echo "[$(date)] lote 19 fim"

echo "[$(date)] TODOS OS LOTES DE TREINO CONCLUIDOS."
