#!/bin/bash
# Re-execucao COMPLETA do conjunto de TREINO (57 videos) COM GS-HOTA.
# Lotes de 3 videos (mesma metodologia do test/valid -> media por lote).
# Por cada lote:  corre -> GS-HOTA para ficheiro PERMANENTE -> metricas JNR/equipa -> APAGA o estado.
# Sem 'set -e': um lote que falhe nao trava os restantes.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
TL=/home/joao/soccernet/venv_tl/bin/tracklab
PY_M=/home/joao/soccernet/venv_metrics/bin/python
SN=/home/joao/soccernet/sn-gamestate
LOGD=$SN/docs/train_logs
RES=$SN/docs/train_gshota.txt
mkdir -p "$LOGD" "$SN/docs/train_reports"
cd $SN

run_lote () {   # $1=NN  $2=lista_virgulas  $3=lista_espacos
  N=$1
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$FREE" -lt 8 ]; then
    echo "lote $N | ABORTADO: apenas ${FREE}G livres em disco" | tee -a "$RES"; return
  fi
  echo "===== LOTE $N : $3  (livre: ${FREE}G) ====="
  echo "[$(date '+%F %T')] lote $N inicio | $3" | tee -a "$RES"
  $TL -cn soccernet \
    dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    "dataset.vids_dict.train=[$2]" \
    state.save_file=states/train-b$N.pklz > "$LOGD/batch$N.log" 2>&1
  G=$(grep -oE "performance GS-HOTA = [0-9.]+" "$LOGD/batch$N.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "lote $N | ${3// /,} | GS-HOTA = ${G:-FALHOU}" | tee -a "$RES"
  S=$(ls -t $SN/outputs/sn-gamestate/*/*/states/train-b$N.pklz 2>/dev/null | head -1)
  if [ -n "$S" ]; then
    $PY_M generate_report.py "$S" --gt_base /home/joao/soccernet/data/SoccerNetGS/train \
        --out docs/train_reports/train-b$N.md >/dev/null 2>&1 \
      && echo "  metricas -> docs/train_reports/train-b$N.md" | tee -a "$RES" \
      || echo "  AVISO: generate_report falhou (estado MANTIDO)" | tee -a "$RES"
    if [ -f "docs/train_reports/train-b$N.md" ]; then rm -f "$S"; fi
  else
    echo "  SEM ESTADO -> lote $N nao produziu resultado" | tee -a "$RES"
  fi
}

echo "[$(date '+%F %T')] INICIO: 57 videos de treino, 19 lotes, com GS-HOTA" | tee -a "$RES"

run_lote 01 "SNGS-060,SNGS-061,SNGS-062" "SNGS-060 SNGS-061 SNGS-062"
run_lote 02 "SNGS-063,SNGS-064,SNGS-065" "SNGS-063 SNGS-064 SNGS-065"
run_lote 03 "SNGS-066,SNGS-067,SNGS-068" "SNGS-066 SNGS-067 SNGS-068"
run_lote 04 "SNGS-069,SNGS-070,SNGS-071" "SNGS-069 SNGS-070 SNGS-071"
run_lote 05 "SNGS-072,SNGS-073,SNGS-074" "SNGS-072 SNGS-073 SNGS-074"
run_lote 06 "SNGS-075,SNGS-076,SNGS-077" "SNGS-075 SNGS-076 SNGS-077"
run_lote 07 "SNGS-097,SNGS-098,SNGS-099" "SNGS-097 SNGS-098 SNGS-099"
run_lote 08 "SNGS-100,SNGS-101,SNGS-102" "SNGS-100 SNGS-101 SNGS-102"
run_lote 09 "SNGS-103,SNGS-104,SNGS-105" "SNGS-103 SNGS-104 SNGS-105"
run_lote 10 "SNGS-106,SNGS-107,SNGS-108" "SNGS-106 SNGS-107 SNGS-108"
run_lote 11 "SNGS-109,SNGS-110,SNGS-111" "SNGS-109 SNGS-110 SNGS-111"
run_lote 12 "SNGS-112,SNGS-113,SNGS-114" "SNGS-112 SNGS-113 SNGS-114"
run_lote 13 "SNGS-115,SNGS-151,SNGS-152" "SNGS-115 SNGS-151 SNGS-152"
run_lote 14 "SNGS-153,SNGS-154,SNGS-155" "SNGS-153 SNGS-154 SNGS-155"
run_lote 15 "SNGS-156,SNGS-157,SNGS-158" "SNGS-156 SNGS-157 SNGS-158"
run_lote 16 "SNGS-159,SNGS-160,SNGS-161" "SNGS-159 SNGS-160 SNGS-161"
run_lote 17 "SNGS-162,SNGS-163,SNGS-164" "SNGS-162 SNGS-163 SNGS-164"
run_lote 18 "SNGS-165,SNGS-166,SNGS-167" "SNGS-165 SNGS-166 SNGS-167"
run_lote 19 "SNGS-168,SNGS-169,SNGS-170" "SNGS-168 SNGS-169 SNGS-170"

echo "[$(date '+%F %T')] TODOS OS LOTES CONCLUIDOS" | tee -a "$RES"
