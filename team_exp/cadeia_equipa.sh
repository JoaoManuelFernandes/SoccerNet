#!/bin/bash
# Cadeia completa depois da auditoria do agrupamento (2026-09-22): refaz só o que a alteração afeta.
set -u
G=/home/joao/soccernet/nosso_out/gshota; N=/home/joao/soccernet/nosso_out; T=/home/joao/soccernet/tese
PYM=/home/joao/soccernet/venv_metrics/bin/python; PYTL=/home/joao/soccernet/venv_tl/bin/python
LOG="$G/equipa_progresso.log"; export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; unset PYTHONPATH
MUD="SNGS-025 SNGS-027 SNGS-028 SNGS-030 SNGS-036 SNGS-038 SNGS-039 SNGS-069 SNGS-091 SNGS-131"
echo "[$(date '+%F %T')] INÍCIO — estados reescritos: $MUD" >> "$LOG"
# 1) apagar os artefactos obsoletos do NOSSO sistema (os da referência ficam intactos)
for V in $MUD; do rm -f "$G"/preds/nosso/SoccerNetGS-*/tracklab/$V.json; done
rm -f "$G"/nosso_*.pklz "$G"/nosso_*.json "$G"/decomp_nosso_*.json
echo "[$(date '+%F %T')] GS-HOTA do nosso (só os 10 vídeos precisam de predições novas)" >> "$LOG"
cd /home/joao/soccernet/sn-gamestate
SYS=nosso bash -c '
G=/home/joao/soccernet/nosso_out/gshota; SN=/home/joao/soccernet/sn-gamestate
PYM=/home/joao/soccernet/venv_metrics/bin/python; PYTL=/home/joao/soccernet/venv_tl/bin/python; TL=/home/joao/soccernet/venv_tl/bin/tracklab
for SPLIT in test valid train; do
  PRED="$G/preds/nosso"; DST="$PRED/SoccerNetGS-$SPLIT/tracklab"; ST="$G/nosso_${SPLIT}.pklz"
  [ -s "$ST" ] || $PYM "$G/junta_estados.py" /home/joao/soccernet/nosso_out/states "$SPLIT" "$ST" > "$G/nosso_${SPLIT}.junta.log" 2>&1
  VIDS=$(unzip -l "$ST" | grep -o "[0-9]\{3\}_image.pkl" | sed "s/_image.pkl//" | sort | sed "s/^/SNGS-/")
  N=$(echo "$VIDS" | wc -l); echo "[$(date "+%F %T")] nosso $SPLIT: $N vídeos" >> "$G/equipa_progresso.log"
  FALTA=$(for V in $VIDS; do [ -s "$DST/$V.json" ] || echo "$V"; done)
  echo "$FALTA" | grep -v "^$" | xargs -r -n 8 | while read -r BLOCO; do
    LISTA=$(echo "$BLOCO" | tr " " ",")
    $TL -cn soccernet num_cores=1 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 "++dataset.vids_dict.$SPLIT=[$LISTA]" \
       eval_tracking=True "pipeline=[]" state.load_file="$ST" state.save_file=null visualization.cfg.save_videos=False > "$G/nosso_${SPLIT}.bloco.log" 2>&1
    D=$(ls -td "$SN"/outputs/sn-gamestate/*/* | head -1); cp "$D"/eval/pred/SoccerNetGS-$SPLIT/tracklab/*.json "$DST"/ 2>/dev/null
    echo "  bloco [$LISTA] -> $(ls "$DST" | wc -l)/$N json" >> "$G/equipa_progresso.log"
    find "$SN/outputs" -mindepth 1 -maxdepth 2 -type d -mmin +2 -exec rm -rf {} + 2>/dev/null
  done
  $PYTL "$G/eval_trackeval.py" "$PRED" "$SPLIT" "$G/nosso_${SPLIT}.json" > "$G/nosso_${SPLIT}.eval.log" 2>&1
  L=$($PYM -c "import json; d=json.load(open(\"$G/nosso_${SPLIT}.json\")); print(f\"| nosso | $SPLIT | {d[chr(39)+chr(39)] if False else d[\"n_videos\"]} | {d[\"GS-HOTA\"]:.2f} | {d[\"DetA\"]:.2f} | {d[\"AssA\"]:.2f} | {d[\"LocA\"]:.2f} |\")" 2>/dev/null || echo "| nosso | $SPLIT | ? | FALHOU | | | |")
  echo "  -> $L" >> "$G/equipa_progresso.log"
done'
# 2) RESUMO_GSHOTA.md: manter as linhas da referência, substituir as do nosso
$PYM - <<'PY' >> "$LOG" 2>&1
import json,re
G="/home/joao/soccernet/nosso_out/gshota"; p=f"{G}/RESUMO_GSHOTA.md"
lines=[l for l in open(p) if not re.match(r"\| nosso \|", l)]
for sp in ("test","valid","train"):
    d=json.load(open(f"{G}/nosso_{sp}.json"))
    lines.append(f"| nosso | {sp} | {d['n_videos']} | {d['GS-HOTA']:.2f} | {d['DetA']:.2f} | {d['AssA']:.2f} | {d['LocA']:.2f} |\n")
open(p,"w").writelines(lines); print("RESUMO_GSHOTA.md atualizado")
PY
echo "[$(date '+%F %T')] decomposição por atributo (só o nosso)" >> "$LOG"
bash "$G/corre_decomposicao.sh" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] comparação por tracklet e tabela" >> "$LOG"
$PYM "$N/tools/compara_final.py" >> "$LOG" 2>&1
$PYM "$N/tools/tabela_video_a_video.py" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] capítulos, figuras e anexo" >> "$LOG"
$PYM "$T/gerador/gera_tese.py" >> "$LOG" 2>&1
$PYTL /home/joao/soccernet/sn-gamestate/docs/figuras_tese/fig_cap4.py >> "$LOG" 2>&1
$PYM "$T/gerador/gera_anexo.py" >> "$LOG" 2>&1
bash "$T/gerador/monta_para_carregar.sh" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] CADEIA DA EQUIPA CONCLUÍDA" >> "$LOG"
