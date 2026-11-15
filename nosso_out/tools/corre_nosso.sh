#!/bin/bash
# Fase 3: a NOSSA pipeline (sn-gamestate com os nossos módulos) nos 160 vídeos, só estados
# (os nossos MP4 já existem em docs/full_report e docs/final_report). Tempos por módulo no log ([TEMPO]).
# Espera a cadeia da baseline acabar. Pára com < 3 GB ou sem CUDA. Retoma onde ficou.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset PYTHONPATH                          # o venv_tl resolve sn_gamestate para /home/joao/soccernet/sn-gamestate (o nosso)
SN=/home/joao/soccernet/sn-gamestate
TL=/home/joao/soccernet/venv_tl/bin/tracklab
PYTL=/home/joao/soccernet/venv_tl/bin/python
PYM=/home/joao/soccernet/venv_metrics/bin/python
OUT=/home/joao/soccernet/nosso_out
LOG="$OUT/progresso.log"
BOUT=/home/joao/soccernet/baseline_out
while systemctl --user is-active baseline-tudo >/dev/null 2>&1; do sleep 120; done
cuda_ok() { for i in 1 2 3; do $PYTL -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1 && return 0; sleep 30; done; return 1; }
disco_ok() { FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9'); [ "$FREE" -ge 3 ]; }
slim() { [ -s "$1" ] && [ "$(stat -c %s "$1")" -gt 30000000 ] && $PYM "$BOUT/tools/slim_estado.py" "$1" "$1" >> "$LOG" 2>&1; }
cd "$SN"
for SPLIT in test valid train; do
  VIDS=$(ls -1 /home/joao/soccernet/data/SoccerNetGS/$SPLIT | grep '^SNGS-' | sort); NT=$(echo "$VIDS" | wc -l); N=0
  for V in $VIDS; do
    N=$((N+1))
    [ -s "$OUT/states/$V.pklz" ] && continue
    disco_ok || { echo "[$(date '+%F %T')] PARADO: disco cheio (<3 GB)" >> "$LOG"; exit 3; }
    cuda_ok  || { echo "[$(date '+%F %T')] PARADO: CUDA indisponível" >> "$LOG"; exit 2; }
    echo "[$(date '+%F %T')] $SPLIT $V ($N/$NT) nosso, livre=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')G" >> "$LOG"
    rm -f "$OUT/states/$V.pklz"
    $TL -cn soccernet num_cores=2 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 \
        "++dataset.vids_dict.$SPLIT=[$V]" eval_tracking=False visualization.cfg.save_videos=False \
        modules.jersey_number_detect.save_debug_crops=false \
        state.load_file=null state.save_file="$OUT/states/$V.pklz" > "$OUT/logs/$V.log" 2>&1
    if [ -s "$OUT/states/$V.pklz" ]; then
      echo "  -> ok, estado $(du -h "$OUT/states/$V.pklz" | cut -f1)" >> "$LOG"; slim "$OUT/states/$V.pklz"
      $PYM "$OUT/tools/compara_final.py" > "$OUT/tools/compara_final.log" 2>&1
    else echo "  -> FALHOU (ver logs/$V.log)" >> "$LOG"; fi
    find "$SN/outputs" -mindepth 1 -maxdepth 2 -type d -mmin +5 -exec rm -rf {} + 2>/dev/null
  done
  echo "[$(date '+%F %T')] SPLIT $SPLIT CONCLUÍDO (nosso)" >> "$LOG"
done
echo "[$(date '+%F %T')] NOSSA PIPELINE COMPLETA (160)" >> "$LOG"
