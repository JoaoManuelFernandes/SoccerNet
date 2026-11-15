#!/bin/bash
# Corre 8 batches de 5 vídeos (fase 1: sem JNR)
# Uso: cd /home/joao/soccernet/sn-gamestate && bash /home/joao/soccernet/run_batches.sh

set -e
source /home/joao/miniconda3/etc/profile.d/conda.sh
conda activate tracklab
cd /home/joao/soccernet/sn-gamestate

YAML=/home/joao/soccernet/sn-gamestate/sn_gamestate/configs/soccernet.yaml
LOG_DIR=/tmp

run_batch() {
  local batch=$1
  local nvid=$2
  local save_file=$3
  shift 3
  local vids=("$@")

  echo ""
  echo "=========================================="
  echo "BATCH $batch — ${#vids[@]} vídeos: ${vids[*]}"
  echo "=========================================="

  # Gerar bloco de vids_dict
  local vids_yaml=""
  for v in "${vids[@]}"; do
    vids_yaml+="      - '$v'"$'\n'
  done

  # Actualizar yaml (dataset + state)
  python3 - <<PYEOF
import re

with open("$YAML") as f:
    content = f.read()

# Substituir nvid
content = re.sub(r'(# BATCH \d+/8[^\n]*\ndataset:\n  nvid: )\d+', r'\g<1>$nvid', content)

# Substituir bloco vids_dict test
content = re.sub(
    r'(  vids_dict:\n    test:\n)((?:      - \'[^\n]+\'\n)+)',
    r'\g<1>$vids_yaml',
    content
)

# Substituir save_file
content = re.sub(r'(  save_file: states/)[^\n]+', r'\g<1>$save_file', content)

# Actualizar label do batch
content = re.sub(r'# BATCH \d+/8', '# BATCH $batch/8', content)

with open("$YAML", "w") as f:
    f.write(content)

print("YAML actualizado: nvid=$nvid, save=$save_file")
PYEOF

  tracklab -cn soccernet 2>&1 | tee "$LOG_DIR/run_batch${batch}.log"
  echo "Batch $batch concluído. State: outputs/.../states/$save_file"
}

# 8 batches × 5 vídeos (último tem 4)
run_batch 1  5 "sn-gamestate-test-batch1.pklz"  SNGS-118 SNGS-119 SNGS-120 SNGS-121 SNGS-122
run_batch 2  5 "sn-gamestate-test-batch2.pklz"  SNGS-123 SNGS-124 SNGS-125 SNGS-126 SNGS-129
run_batch 3  5 "sn-gamestate-test-batch3.pklz"  SNGS-132 SNGS-133 SNGS-134 SNGS-135 SNGS-136
run_batch 4  5 "sn-gamestate-test-batch4.pklz"  SNGS-137 SNGS-138 SNGS-139 SNGS-140 SNGS-141
run_batch 5  5 "sn-gamestate-test-batch5.pklz"  SNGS-142 SNGS-145 SNGS-146 SNGS-147 SNGS-148
run_batch 6  5 "sn-gamestate-test-batch6.pklz"  SNGS-149 SNGS-150 SNGS-187 SNGS-188 SNGS-189
run_batch 7  5 "sn-gamestate-test-batch7.pklz"  SNGS-192 SNGS-193 SNGS-194 SNGS-195 SNGS-196
run_batch 8  4 "sn-gamestate-test-batch8.pklz"  SNGS-197 SNGS-198 SNGS-199 SNGS-200

echo ""
echo "=========================================="
echo "TODOS OS BATCHES CONCLUÍDOS!"
echo "State files em: outputs/sn-gamestate/YYYY-MM-DD/HH-MM-SS/states/"
echo "=========================================="
