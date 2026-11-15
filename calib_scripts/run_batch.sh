#!/bin/bash
# Actualiza o soccernet.yaml para o lote pedido, lança o pipeline e espera pelo GS-HOTA.
# Uso: run_batch.sh <NOME_LOTE> <eval_set> <vid1> [vid2] [vid3]
# Ex.:  run_batch.sh valid_V17 valid SNGS-091 SNGS-092 SNGS-093
set -u
LOTE="$1"; EVALSET="$2"; shift 2
VIDS=("$@")
LOG="/tmp/gshota_${LOTE}.log"
YAML="/home/joao/soccernet/sn-gamestate/sn_gamestate/configs/soccernet.yaml"

# Actualizar yaml (dataset + save_file) via python
python3 - "$YAML" "$EVALSET" "$LOTE" "${VIDS[@]}" <<'EOF'
import re, sys
yaml_path, evalset, lote = sys.argv[1], sys.argv[2], sys.argv[3]
vids = sys.argv[4:]
text = open(yaml_path).read()

vid_lines = "\n".join(f"      - '{v}'" for v in vids)
new_dataset = f"""# Lote {lote} (gerado por run_batch.sh)
dataset:
  nvid: {len(vids)}
  nframes: -1
  eval_set: "{evalset}"
  dataset_path: ${{data_dir}}/SoccerNetGS
  vids_dict:
    {evalset}:
{vid_lines}

# Tracker state — sem load_file (pipeline do zero)
state:
  load_file: null
  save_file: states/gshota-{lote.replace('_','-')}.pklz"""

# Substitui o bloco desde o comentário do lote (ou "dataset:") até ao fim de "state:...save_file: ..."
pattern = re.compile(r"#[^\n]*[Ll]ote[^\n]*\ndataset:.*?save_file:[^\n]*", re.S)
if not pattern.search(text):
    pattern = re.compile(r"dataset:\n  nvid:.*?save_file:[^\n]*", re.S)
text = pattern.sub(new_dataset, text, count=1)
open(yaml_path, "w").write(text)
print(f"[yaml] lote={lote} eval_set={evalset} vids={vids}")
EOF

cd /home/joao/soccernet/sn-gamestate
conda run -n tracklab --no-capture-output tracklab -cn soccernet > "$LOG" 2>&1 &
PID=$!

while kill -0 "$PID" 2>/dev/null; do
  if grep -q "GS-HOTA =\|Error executing\|Traceback" "$LOG" 2>/dev/null; then
    break
  fi
  sleep 15
done
wait "$PID" 2>/dev/null

echo "=== $LOTE TERMINADO ==="
grep -A3 "GS-HOTA =" "$LOG" | head -5
grep -A2 "HOTA: tracklab-cls_comb_cls_av" "$LOG" | head -3
