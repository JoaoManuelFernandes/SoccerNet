#!/bin/bash
# Opção A — passagem FINAL: pipeline completo (Qwen + team-side recalibrado),
# GS-HOTA + métricas por vídeo + vídeo de visualização final, tudo numa corrida.
# Substitui os vídeos antigos (mesmo nome) para o disco não crescer.
#
# Uso: gen_final.sh <LOTE> <eval_set> <load_file|none> <vid1> [vid2...]
#   test : load_file = *-full.pklz (tem bbox/track/calib/role) → 4 módulos
#   valid: load_file = none → pipeline completo do zero
set -u
LOTE="$1"; EVALSET="$2"; LOADFILE="$3"; shift 3
VIDS=("$@")
LOG="/tmp/final_${LOTE}.log"
BASE="/home/joao/soccernet/sn-gamestate"
CONFIGS="$BASE/sn_gamestate/configs"
VIDOUT="$BASE/docs/final_report/videos/$EVALSET"
METOUT="$BASE/docs/final_report/metrics"
OLDVID="$BASE/docs/full_report/videos/$EVALSET"
mkdir -p "$VIDOUT" "$METOUT"

python3 - "$CONFIGS" "$EVALSET" "$LOTE" "$LOADFILE" "${VIDS[@]}" <<'EOF'
import re, sys
configs, evalset, lote, loadfile = sys.argv[1:5]
vids = sys.argv[5:]
text = open(f"{configs}/soccernet.yaml").read()

if loadfile != "none":
    new_pipeline = ("pipeline:\n  - team\n  - team_side\n"
                    "  - jersey_number_detect\n  - tracklet_agg\n")
    load_line = f'load_file: "{loadfile}"'
else:
    new_pipeline = ("pipeline:\n  - bbox_detector\n  - reid\n  - track\n"
                    "  - pitch\n  - calibration\n  - jersey_number_detect\n"
                    "  - tracklet_agg\n  - team\n  - team_side\n")
    load_line = "load_file: null"
text = re.sub(r"pipeline:(?:\s*\[\]|\n(?:  - \w+\n)+)", new_pipeline, text, count=1)
# tracklet_agg com jersey+role (default voting_role_jn) — garantir que NÃO ficou role_only
text = text.replace("modules/tracklet_agg: voting_role_only",
                    "modules/tracklet_agg: voting_role_jn")

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
  save_file: states/final-{lote}.pklz"""
pattern = re.compile(r"(?:#[^\n]*\n)?dataset:\n  nvid:.*?save_file:[^\n]*", re.S)
text = pattern.sub(new_dataset, text, count=1)
text = text.replace("save_videos: False", "save_videos: True")
text = re.sub(r"eval_tracking: (True|False)[^\n]*", "eval_tracking: True", text)
open(f"{configs}/soccernet_final.yaml", "w").write(text)
print(f"[yaml] {lote} load={loadfile!='none'} vids={vids}")
EOF

cd "$BASE"
conda run -n tracklab --no-capture-output tracklab -cn soccernet_final > "$LOG" 2>&1
echo "PIPELINE_RC=$?"

echo "--- GS-HOTA ---"
grep -A2 "GS-HOTA =" "$LOG" | head -3

RUNDIR=$(ls -dt "$BASE"/outputs/sn-gamestate/*/*/ 2>/dev/null | head -1)
PKLZ="${RUNDIR}states/final-${LOTE}.pklz"
if [ -f "$PKLZ" ]; then
  conda run -n tracklab --no-capture-output python3 "$BASE/generate_report.py" "$PKLZ" --out "$METOUT/${LOTE}.md" >> "$LOG" 2>&1
  echo "METRICS_RC=$?"
  MOVED=0
  for f in "$RUNDIR"visualization/videos/*.mp4; do
    [ -f "$f" ] || continue
    bn=$(basename "$f" .mp4)
    mv -f "$f" "$VIDOUT/SNGS-${bn}.mp4"
    rm -f "$OLDVID/SNGS-${bn}.mp4"   # remove versão antiga (pré-calibração)
    MOVED=$((MOVED+1))
  done
  echo "VIDEOS_FINAIS=$MOVED"
  rm -rf "$RUNDIR"
  echo "LIMPO run dir"
else
  echo "ERRO: sem pklz"; tail -15 "$LOG"
fi
df -h / | tail -1
echo "=== $LOTE FINAL DONE ==="
