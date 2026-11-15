#!/bin/bash
# Experiência JNR (A/C) — corre pipeline do zero em vídeos escolhidos, guarda
# pklz + jnr_v2_debug (historial+window_meta) + report JNR, numa pasta ISOLADA.
# NÃO toca em docs/final_report nem no relatório.
# Uso: gen_jnrexp.sh <TAG> <eval_set> <clahe:0|1> <vid1> [vid2...]
set -u
TAG="$1"; EVALSET="$2"; CLAHE="$3"; shift 3
VIDS=("$@")
LOG="/tmp/jnrexp_${TAG}.log"
BASE="/home/joao/soccernet/sn-gamestate"
CONFIGS="$BASE/sn_gamestate/configs"
EXPROOT="/home/joao/soccernet/jnr_exp/$TAG"
mkdir -p "$EXPROOT"

python3 - "$CONFIGS" "$EVALSET" "$TAG" "$CLAHE" "${VIDS[@]}" <<'EOF'
import re, sys
configs, evalset, tag, clahe = sys.argv[1:5]
vids = sys.argv[5:]
text = open(f"{configs}/soccernet.yaml").read()
new_pipeline = ("pipeline:\n  - bbox_detector\n  - reid\n  - track\n"
                "  - pitch\n  - calibration\n  - jersey_number_detect\n  - tracklet_agg\n")
text = re.sub(r"pipeline:(?:\s*\[\]|\n(?:  - \w+\n)+)", new_pipeline, text, count=1)
text = text.replace("modules/tracklet_agg: voting_role_only",
                    "modules/tracklet_agg: voting_role_jn")
# módulo jnr: debug ON + clahe conforme flag
variant = clahe  # 0=base, 2=montage (clahe removido do código)
extra = "qwen_montage: true, " if variant == "2" else ""
text = re.sub(r"jersey_number_detect: \{[^}]*\}",
              f"jersey_number_detect: {{batch_size: 4, save_debug_crops: true, "
              f"{extra}max_tracklets_per_qwen_batch: 4}}",
              text, count=1)
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
  save_file: states/jnrexp-{tag}.pklz"""
pattern = re.compile(r"(?:#[^\n]*\n)?dataset:\n  nvid:.*?save_file:[^\n]*", re.S)
text = pattern.sub(new_dataset, text, count=1)
text = text.replace("save_videos: True", "save_videos: False")
text = re.sub(r"eval_tracking: (True|False)[^\n]*", "eval_tracking: False", text)
open(f"{configs}/soccernet_jnrexp.yaml", "w").write(text)
print(f"[yaml] tag={tag} clahe={clahe_bool} vids={vids}")
EOF

cd "$BASE"
conda run -n tracklab --no-capture-output tracklab -cn soccernet_jnrexp > "$LOG" 2>&1
echo "PIPELINE_RC=$?"

RUNDIR=$(ls -dt "$BASE"/outputs/sn-gamestate/*/*/ 2>/dev/null | head -1)
PKLZ="${RUNDIR}states/jnrexp-${TAG}.pklz"
if [ -f "$PKLZ" ]; then
  cp "$PKLZ" "$EXPROOT/"
  # copiar jnr_v2_debug (historial + window_meta) se existir
  [ -d "${RUNDIR}jnr_v2_debug" ] && cp -r "${RUNDIR}jnr_v2_debug" "$EXPROOT/" 2>/dev/null
  conda run -n tracklab --no-capture-output python3 "$BASE/generate_report.py" "$PKLZ" --out "$EXPROOT/report.md" >> "$LOG" 2>&1
  echo "REPORT_RC=$?"
  echo "--- JNR por vídeo ($TAG) ---"
  python3 -c "
import re
for m in re.finditer(r'### (SNGS-\d+)(.*?)(?=### SNGS-|\Z)', open('$EXPROOT/report.md').read(), re.S):
    t=re.search(r'#### JNR.*?\|\s*(\d+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*\*\*([\d.]+)%\*\*',m.group(2),re.S)
    if t: print(f'  {m.group(1)}: {t.group(3)}% ({t.group(1) and t.group(1)}/{t.group(2)})')
"
  rm -rf "$RUNDIR"
  echo "LIMPO run dir (pklz+debug preservados em $EXPROOT)"
else
  echo "ERRO: sem pklz"; tail -15 "$LOG"
fi
df -h / | tail -1
echo "=== $TAG JNREXP DONE ==="
