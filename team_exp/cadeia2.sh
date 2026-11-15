#!/bin/bash
set -u
G=/home/joao/soccernet/nosso_out/gshota; N=/home/joao/soccernet/nosso_out; T=/home/joao/soccernet/tese; SN=/home/joao/soccernet/sn-gamestate
PYM=/home/joao/soccernet/venv_metrics/bin/python; PYTL=/home/joao/soccernet/venv_tl/bin/python; TL=/home/joao/soccernet/venv_tl/bin/tracklab
LOG="$G/equipa_progresso.log"; export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; unset PYTHONPATH
echo "[$(date '+%F %T')] 2.ª passagem: guardas para 039 (árbitros) e 069 (resgate degenerado)" >> "$LOG"
cd "$SN"
for SPLIT in valid train; do
  PRED="$G/preds/nosso"; DST="$PRED/SoccerNetGS-$SPLIT/tracklab"; ST="$G/nosso_${SPLIT}.pklz"
  [ -s "$ST" ] || $PYM "$G/junta_estados.py" "$N/states" "$SPLIT" "$ST" > "$G/nosso_${SPLIT}.junta.log" 2>&1
  VIDS=$(unzip -l "$ST" | grep -o "[0-9]\{3\}_image.pkl" | sed "s/_image.pkl//" | sort | sed "s/^/SNGS-/")
  N2=$(echo "$VIDS" | wc -l)
  FALTA=$(for V in $VIDS; do [ -s "$DST/$V.json" ] || echo "$V"; done)
  echo "$FALTA" | grep -v "^$" | xargs -r -n 8 | while read -r BLOCO; do
    LISTA=$(echo "$BLOCO" | tr " " ",")
    $TL -cn soccernet num_cores=1 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 "++dataset.vids_dict.$SPLIT=[$LISTA]" \
       eval_tracking=True "pipeline=[]" state.load_file="$ST" state.save_file=null visualization.cfg.save_videos=False > "$G/nosso_${SPLIT}.bloco.log" 2>&1
    D=$(ls -td "$SN"/outputs/sn-gamestate/*/* | head -1); cp "$D"/eval/pred/SoccerNetGS-$SPLIT/tracklab/*.json "$DST"/ 2>/dev/null
    echo "  bloco [$LISTA] -> $(ls "$DST" | wc -l)/$N2 json" >> "$LOG"
    find "$SN/outputs" -mindepth 1 -maxdepth 2 -type d -mmin +2 -exec rm -rf {} + 2>/dev/null
  done
  $PYTL "$G/eval_trackeval.py" "$PRED" "$SPLIT" "$G/nosso_${SPLIT}.json" > "$G/nosso_${SPLIT}.eval.log" 2>&1
  echo "  -> $($PYM -c "import json;d=json.load(open('$G/nosso_${SPLIT}.json'));print(f\"| nosso | $SPLIT | {d['n_videos']} | {d['GS-HOTA']:.2f} | {d['DetA']:.2f} | {d['AssA']:.2f} | {d['LocA']:.2f} |\")")" >> "$LOG"
done
$PYM - <<'PY' >> "$LOG" 2>&1
import json,re
G="/home/joao/soccernet/nosso_out/gshota"; p=f"{G}/RESUMO_GSHOTA.md"
lines=[l for l in open(p) if not re.match(r"\| nosso \| (valid|train) \|", l)]
for sp in ("valid","train"):
    d=json.load(open(f"{G}/nosso_{sp}.json"))
    lines.append(f"| nosso | {sp} | {d['n_videos']} | {d['GS-HOTA']:.2f} | {d['DetA']:.2f} | {d['AssA']:.2f} | {d['LocA']:.2f} |\n")
open(p,"w").writelines(lines)
PY
bash "$G/corre_decomposicao.sh" >> "$LOG" 2>&1
$PYM "$N/tools/compara_final.py" >> "$LOG" 2>&1
$PYM "$N/tools/tabela_video_a_video.py" >> "$LOG" 2>&1
$PYM "$T/gerador/gera_tese.py" >> "$LOG" 2>&1
$PYTL "$SN/docs/figuras_tese/fig_cap4.py" >> "$LOG" 2>&1
$PYM "$T/gerador/gera_anexo.py" >> "$LOG" 2>&1
bash "$T/gerador/monta_para_carregar.sh" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] 2.ª PASSAGEM CONCLUÍDA" >> "$LOG"
