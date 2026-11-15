#!/bin/bash
# Pipeline RÁPIDO (sem Qwen/JNR) para calibração team-side no valid:
# corre bbox→reid→track→pitch→calib→agg→team→team_side, extrai dump de
# calibração do pklz e apaga tudo.
# Uso: run_team_batch.sh <LOTE> <eval_set> <vid1> [vid2] [vid3]
set -u
LOTE="$1"; EVALSET="$2"; shift 2
VIDS=("$@")
LOG="/tmp/teamcalib_${LOTE}.log"
BASE="/home/joao/soccernet/sn-gamestate"
CONFIGS="$BASE/sn_gamestate/configs"
S="/home/joao/soccernet/calib_scripts"
DUMPDIR="/home/joao/soccernet/calib_dumps/$EVALSET"

python3 - "$CONFIGS" "$EVALSET" "$LOTE" "${VIDS[@]}" <<'EOF'
import re, sys
configs, evalset, lote = sys.argv[1], sys.argv[2], sys.argv[3]
vids = sys.argv[4:]
text = open(f"{configs}/soccernet.yaml").read()

# pipeline sem jersey_number_detect (Qwen) — só o necessário para team-side
new_pipeline = """pipeline:
  - bbox_detector
  - reid
  - track
  - pitch
  - calibration
  - tracklet_agg
  - team
  - team_side
"""
text = re.sub(r"pipeline:(?:\s*\[\]|\n(?:  - \w+\n)+)", new_pipeline, text, count=1)

# tracklet_agg só com role (sem JNR no pipeline não há colunas jersey_number_detection)
text = text.replace("modules/tracklet_agg: voting_role_jn",
                    "modules/tracklet_agg: voting_role_only")

vid_lines = "\n".join(f"      - '{v}'" for v in vids)
new_dataset = f"""dataset:
  nvid: {len(vids)}
  nframes: -1
  eval_set: "{evalset}"
  dataset_path: ${{data_dir}}/SoccerNetGS
  vids_dict:
    {evalset}:
{vid_lines}

# Tracker state
state:
  load_file: null
  save_file: states/teamcalib-{lote}.pklz"""
pattern = re.compile(r"(?:#[^\n]*\n)?dataset:\n  nvid:.*?save_file:[^\n]*", re.S)
text = pattern.sub(new_dataset, text, count=1)

text = text.replace("save_videos: True", "save_videos: False")
text = re.sub(r"eval_tracking: (True|False)[^\n]*", "eval_tracking: False", text)

open(f"{configs}/soccernet_teamcalib.yaml", "w").write(text)
print(f"[yaml] {lote} vids={vids}")
EOF

cd "$BASE"
conda run -n tracklab --no-capture-output tracklab -cn soccernet_teamcalib > "$LOG" 2>&1
RC=$?
echo "PIPELINE_RC=$RC"

RUNDIR=$(ls -dt "$BASE"/outputs/sn-gamestate/*/*/ 2>/dev/null | head -1)
PKLZ="${RUNDIR}states/teamcalib-${LOTE}.pklz"
if [ -f "$PKLZ" ]; then
  conda run -n tracklab --no-capture-output python3 "$S/dump_calib.py" "$PKLZ" "$DUMPDIR"
  echo "DUMP_RC=$?"
  N=$(ls "$DUMPDIR" | wc -l)
  echo "DUMPS_TOTAL=$N"
  rm -rf "$RUNDIR"
  echo "LIMPO: $RUNDIR"
else
  echo "ERRO: pklz não encontrado em $PKLZ"
  tail -20 "$LOG"
fi
df -h / | tail -1
echo "=== $LOTE TEAMCALIB DONE ==="
