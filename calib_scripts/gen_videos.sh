#!/bin/bash
# Gera vídeos de visualização + métricas por vídeo a partir de um pklz existente.
# Uso: gen_videos.sh <pklz_path> <lote> <eval_set> <vid1> [vid2] [vid3]
# Depois de extrair tudo, apaga o run dir E o pklz de origem (liberta disco).
set -u
PKLZ="$1"; LOTE="$2"; EVALSET="$3"; shift 3
VIDS=("$@")
BASE="/home/joao/soccernet/sn-gamestate"
CONFIGS="$BASE/sn_gamestate/configs"
DOCS="$BASE/docs/full_report"
LOG="/tmp/genvid_${LOTE}.log"

mkdir -p "$DOCS/videos/$EVALSET" "$DOCS/metrics"

if [ ! -f "$PKLZ" ]; then echo "ERRO: pklz não existe: $PKLZ"; exit 1; fi

# Gera config de visualização a partir do soccernet.yaml
python3 - "$CONFIGS" "$PKLZ" "$EVALSET" "${VIDS[@]}" <<'EOF'
import re, sys
configs, pklz, evalset = sys.argv[1], sys.argv[2], sys.argv[3]
vids = sys.argv[4:]
text = open(f"{configs}/soccernet.yaml").read()

# pipeline vazio (só visualização, nada recomputado)
text = re.sub(r"pipeline:\n(?:  - \w+\n)+", "pipeline: []\n", text, count=1)

# dataset: vids do lote
vid_lines = "\n".join(f"      - '{v}'" for v in vids)
new_dataset = f"""dataset:
  nvid: {len(vids)}
  nframes: -1
  eval_set: "{evalset}"
  dataset_path: ${{data_dir}}/SoccerNetGS
  vids_dict:
    {evalset}:
{vid_lines}

# Tracker state — carrega pklz, não grava nada
state:
  load_file: "{pklz}"
  save_file: null"""
pattern = re.compile(r"(?:#[^\n]*\n)?dataset:\n  nvid:.*?save_file:[^\n]*", re.S)
text = pattern.sub(new_dataset, text, count=1)

# vídeos ligados, avaliação desligada
text = text.replace("save_videos: False", "save_videos: True")
text = text.replace("eval_tracking: True", "eval_tracking: False")
text = re.sub(r"eval_tracking: True[^\n]*", "eval_tracking: False", text)

open(f"{configs}/soccernet_vis.yaml", "w").write(text)
print(f"[vis-yaml] lote={sys.argv[0]} vids={vids}")
EOF

cd "$BASE"
conda run -n tracklab --no-capture-output tracklab -cn soccernet_vis > "$LOG" 2>&1
RC=$?

# Localiza o run dir mais recente (o desta corrida)
RUNDIR=$(ls -dt "$BASE"/outputs/sn-gamestate/*/*/ 2>/dev/null | head -1)
echo "RC=$RC RUNDIR=$RUNDIR"

# Move vídeos
MOVED=0
for f in "$RUNDIR"visualization/videos/*.mp4; do
  [ -f "$f" ] || continue
  bn=$(basename "$f" .mp4)
  mv "$f" "$DOCS/videos/$EVALSET/SNGS-${bn}.mp4"
  MOVED=$((MOVED+1))
done
echo "VIDEOS_MOVIDOS=$MOVED"

# Métricas por vídeo
conda run -n tracklab --no-capture-output python3 "$BASE/generate_report.py" "$PKLZ" --out "$DOCS/metrics/${LOTE}.md" >> "$LOG" 2>&1
echo "METRICS_RC=$?"

# Limpeza: run dir + pklz de origem (só se os vídeos foram extraídos)
if [ "$MOVED" -gt 0 ] && [ -s "$DOCS/metrics/${LOTE}.md" ]; then
  rm -rf "$RUNDIR"
  PKLZ_DIR=$(dirname "$(dirname "$PKLZ")")
  rm -rf "$PKLZ_DIR"
  echo "LIMPO: $RUNDIR e $PKLZ_DIR"
else
  echo "AVISO: extração incompleta, NADA apagado"
fi
df -h / | tail -1
echo "=== $LOTE VÍDEOS+MÉTRICAS DONE ==="
