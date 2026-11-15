#!/bin/bash
# Valida os novos thresholds team-side com o código REAL.
# Test: carrega full.pklz, corre só [team, team_side], avalia, apaga.
# Uso: validate_team.sh <LOTE> <eval_set> <load_file|none> <vid1> [vid2...]
set -u
LOTE="$1"; EVALSET="$2"; LOADFILE="$3"; shift 3
VIDS=("$@")
LOG="/tmp/valteam_${LOTE}.log"
BASE="/home/joao/soccernet/sn-gamestate"
CONFIGS="$BASE/sn_gamestate/configs"
OUTDIR="$BASE/docs/team_recalib"
mkdir -p "$OUTDIR"

python3 - "$CONFIGS" "$EVALSET" "$LOTE" "$LOADFILE" "${VIDS[@]}" <<'EOF'
import re, sys
configs, evalset, lote, loadfile = sys.argv[1:5]
vids = sys.argv[5:]
text = open(f"{configs}/soccernet.yaml").read()

if loadfile != "none":
    new_pipeline = "pipeline:\n  - team\n  - team_side\n"
    load_line = f'load_file: "{loadfile}"'
else:
    new_pipeline = ("pipeline:\n  - bbox_detector\n  - reid\n  - track\n"
                    "  - pitch\n  - calibration\n  - tracklet_agg\n"
                    "  - team\n  - team_side\n")
    load_line = "load_file: null"
text = re.sub(r"pipeline:(?:\s*\[\]|\n(?:  - \w+\n)+)", new_pipeline, text, count=1)
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
  {load_line}
  save_file: states/valteam-{lote}.pklz"""
pattern = re.compile(r"(?:#[^\n]*\n)?dataset:\n  nvid:.*?save_file:[^\n]*", re.S)
text = pattern.sub(new_dataset, text, count=1)
text = text.replace("save_videos: True", "save_videos: False")
text = re.sub(r"eval_tracking: (True|False)[^\n]*", "eval_tracking: False", text)
open(f"{configs}/soccernet_valteam.yaml", "w").write(text)
print(f"[yaml] {lote} load={loadfile!=('none')} vids={vids}")
EOF

cd "$BASE"
conda run -n tracklab --no-capture-output tracklab -cn soccernet_valteam > "$LOG" 2>&1
RC=$?
echo "PIPELINE_RC=$RC"

RUNDIR=$(ls -dt "$BASE"/outputs/sn-gamestate/*/*/ 2>/dev/null | head -1)
PKLZ="${RUNDIR}states/valteam-${LOTE}.pklz"
if [ -f "$PKLZ" ]; then
  conda run -n tracklab --no-capture-output python3 "$BASE/generate_report.py" "$PKLZ" --out "$OUTDIR/${LOTE}.md" >> "$LOG" 2>&1
  echo "METRICS_RC=$?"
  grep -A4 "#### Team" "$OUTDIR/${LOTE}.md" | grep -E "^\|" | tail -40
  rm -rf "$RUNDIR"
  echo "LIMPO"
else
  echo "ERRO: sem pklz"
  tail -15 "$LOG"
fi
echo "=== $LOTE VALTEAM DONE ==="
