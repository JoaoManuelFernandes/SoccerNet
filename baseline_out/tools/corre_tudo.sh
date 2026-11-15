#!/bin/bash
# Cadeia completa da baseline original, sequencial (uma coisa pesada de cada vez):
#   para cada split (test, valid, train) e cada vídeo:
#     1) pipeline original -> estado gordo + vídeo renderizado   (se não houver vídeo)
#     2) passagem OCR original (MMOCR) a partir do estado         (se não houver states_ocr)
#     3) slim dos dois estados no próprio sítio (sem embeddings/máscaras; ~8 MB cada)
#     4) tabela COMPARACAO_POR_VIDEO.md
# Pára com < 3 GB livres (o João avalia e apaga vídeos; relançar com lanca_tudo.sh)
# ou se a CUDA não estiver disponível (depois de uma suspensão: rmmod/modprobe nvidia_uvm).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH=/home/joao/baseline_soccernet/sn-gamestate
BL=/home/joao/baseline_soccernet/sn-gamestate
TL=/home/joao/soccernet/venv_tl/bin/tracklab
PYTL=/home/joao/soccernet/venv_tl/bin/python
PYM=/home/joao/soccernet/venv_metrics/bin/python
OUT=/home/joao/soccernet/baseline_out
LOG="$OUT/progresso.log"
mkdir -p "$OUT/states" "$OUT/videos" "$OUT/logs" "$OUT/states_ocr" "$OUT/logs_ocr"

# espera a corrida antiga do teste acabar, se ainda estiver a correr
while systemctl --user is-active baseline-test >/dev/null 2>&1; do sleep 120; done

cuda_ok() {  # 3 tentativas com 30 s de intervalo: a GPU pode estar a libertar o processo anterior
  for i in 1 2 3; do
    $PYTL -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1 && return 0
    sleep 30
  done
  return 1
}
disco_ok() { FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9'); [ "$FREE" -ge 3 ]; }
slim() { [ -s "$1" ] && [ "$(stat -c %s "$1")" -gt 30000000 ] && $PYM "$OUT/tools/slim_estado.py" "$1" "$1" >> "$LOG" 2>&1; }  # só estados gordos (>30 MB)

cd "$BL"
for SPLIT in test valid train; do
  VIDS=$(ls -1 /home/joao/soccernet/data/SoccerNetGS/$SPLIT | grep '^SNGS-' | sort)
  NT=$(echo "$VIDS" | wc -l); N=0
  for V in $VIDS; do
    N=$((N+1)); MUDOU=0
    # 1) baseline original
    if [ ! -s "$OUT/states/$V.pklz" ]; then
      MUDOU=1   # o estado é a marca de "feito"; o MP4 pode ter sido apagado após avaliação
      disco_ok || { echo "[$(date '+%F %T')] PARADO: disco cheio (<3 GB). Avaliar e apagar vídeos, depois lanca_tudo.sh" >> "$LOG"; exit 3; }
      cuda_ok  || { echo "[$(date '+%F %T')] PARADO: CUDA indisponível (rmmod/modprobe nvidia_uvm)" >> "$LOG"; exit 2; }
      echo "[$(date '+%F %T')] $SPLIT $V ($N/$NT) baseline, livre=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')G" >> "$LOG"
      rm -f "$OUT/states/$V.pklz" "$OUT/videos/$V.mp4"
      $TL -cn soccernet num_cores=3 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 \
          "dataset.vids_dict.$SPLIT=[$V]" eval_tracking=False visualization.cfg.save_videos=True \
          state.save_file="$OUT/states/$V.pklz" > "$OUT/logs/$V.log" 2>&1
      M=$(ls -t "$BL"/outputs/*/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
      if [ -n "$M" ] && [ -s "$OUT/states/$V.pklz" ]; then mv "$M" "$OUT/videos/$V.mp4"
        echo "  -> ok, vídeo $(du -h "$OUT/videos/$V.mp4"|cut -f1), estado $(du -h "$OUT/states/$V.pklz"|cut -f1)" >> "$LOG"
      else echo "  -> FALHOU (ver logs/$V.log)" >> "$LOG"; rm -f "$OUT/states/$V.pklz"; fi
      find "$BL/outputs" -mindepth 1 -maxdepth 2 -type d -mmin +5 -exec rm -rf {} + 2>/dev/null
    fi
    # 2) OCR original a partir do estado (só se o estado ainda tiver embeddings não é preciso; basta existir)
    if [ -s "$OUT/states/$V.pklz" ] && [ ! -s "$OUT/states_ocr/$V.pklz" ]; then
      cuda_ok || { echo "[$(date '+%F %T')] PARADO: CUDA indisponível" >> "$LOG"; exit 2; }
      echo "[$(date '+%F %T')] $SPLIT $V OCR original" >> "$LOG"
      "$OUT/tools/passagem_ocr.sh" "$SPLIT" "$V" >> "$LOG" 2>&1; MUDOU=1
    fi
    # 3) slim (idempotente: um estado já slim fica igual)
    if [ -s "$OUT/states_ocr/$V.pklz" ]; then slim "$OUT/states/$V.pklz"; slim "$OUT/states_ocr/$V.pklz"; fi
    # 4) tabelas (só quando este vídeo mudou; com cache incremental)
    [ "$MUDOU" -eq 1 ] && $PYM "$OUT/tools/compara_por_video.py" > "$OUT/tools/compara_loop.log" 2>&1
  done
  echo "[$(date '+%F %T')] SPLIT $SPLIT CONCLUÍDO" >> "$LOG"
done
# Repetição dos vídeos que falharam na pipeline (ex.: SNGS-096, a calibração original rebenta com
# batch_size=8 num frame sem pontos; com batch_size=1 passa, e o cálculo por imagem é o mesmo).
for SPLIT in test valid train; do
  for V in $(ls -1 /home/joao/soccernet/data/SoccerNetGS/$SPLIT | grep '^SNGS-' | sort); do
    [ -s "$OUT/states/$V.pklz" ] && continue
    disco_ok || exit 3; cuda_ok || exit 2
    echo "[$(date '+%F %T')] $SPLIT $V REPETIÇÃO com calibration.batch_size=1" >> "$LOG"
    rm -f "$OUT/states/$V.pklz" "$OUT/videos/$V.mp4"
    $TL -cn soccernet num_cores=3 modules.calibration.batch_size=1 dataset.eval_set=$SPLIT dataset.nvid=-1 dataset.nframes=-1 \
        "dataset.vids_dict.$SPLIT=[$V]" eval_tracking=False visualization.cfg.save_videos=True \
        state.save_file="$OUT/states/$V.pklz" > "$OUT/logs/$V.log" 2>&1
    M=$(ls -t "$BL"/outputs/*/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
    if [ -n "$M" ] && [ -s "$OUT/states/$V.pklz" ]; then mv "$M" "$OUT/videos/$V.mp4"; echo "  -> ok (repetição)" >> "$LOG"
      "$OUT/tools/passagem_ocr.sh" "$SPLIT" "$V" >> "$LOG" 2>&1
      slim "$OUT/states/$V.pklz"; slim "$OUT/states_ocr/$V.pklz"
      $PYM "$OUT/tools/compara_por_video.py" > "$OUT/tools/compara_loop.log" 2>&1
    else echo "  -> FALHOU outra vez (ver logs/$V.log)" >> "$LOG"; rm -f "$OUT/states/$V.pklz"; fi
    find "$BL/outputs" -mindepth 1 -maxdepth 2 -type d -mmin +5 -exec rm -rf {} + 2>/dev/null
  done
done
echo "[$(date '+%F %T')] CADEIA COMPLETA (160 vídeos)" >> "$LOG"
