#!/bin/bash
# Renderiza os videos do conjunto de TREINO (visualizacao com minimap).
# Ordem: primeiro os 9 com falha de atribuicao de equipa, depois os restantes.
# Sem GS-HOTA (eval_tracking=False) porque as metricas ja estao calculadas.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
TL=/home/joao/soccernet/venv_tl/bin/tracklab
SN=/home/joao/soccernet/sn-gamestate
DEST=$SN/docs/full_report/videos/train
LOGD=$SN/docs/render_logs
mkdir -p "$DEST" "$LOGD"
cd $SN
echo "[$(date)] INICIO da renderizacao do treino" >> $SN/docs/render.log

# --- 1/57 SNGS-106 (PRIORIDADE) ---
if [ ! -f "$DEST/SNGS-106.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-106 (1/57, PRIORIDADE) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-106]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-106.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-106.mp4"; echo "  -> $DEST/SNGS-106.mp4 ($(du -h "$DEST/SNGS-106.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-106" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 2/57 SNGS-111 (PRIORIDADE) ---
if [ ! -f "$DEST/SNGS-111.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-111 (2/57, PRIORIDADE) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-111]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-111.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-111.mp4"; echo "  -> $DEST/SNGS-111.mp4 ($(du -h "$DEST/SNGS-111.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-111" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 3/57 SNGS-113 (PRIORIDADE) ---
if [ ! -f "$DEST/SNGS-113.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-113 (3/57, PRIORIDADE) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-113]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-113.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-113.mp4"; echo "  -> $DEST/SNGS-113.mp4 ($(du -h "$DEST/SNGS-113.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-113" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 4/57 SNGS-069 (PRIORIDADE) ---
if [ ! -f "$DEST/SNGS-069.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-069 (4/57, PRIORIDADE) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-069]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-069.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-069.mp4"; echo "  -> $DEST/SNGS-069.mp4 ($(du -h "$DEST/SNGS-069.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-069" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 5/57 SNGS-064 (PRIORIDADE) ---
if [ ! -f "$DEST/SNGS-064.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-064 (5/57, PRIORIDADE) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-064]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-064.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-064.mp4"; echo "  -> $DEST/SNGS-064.mp4 ($(du -h "$DEST/SNGS-064.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-064" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 6/57 SNGS-068 (PRIORIDADE) ---
if [ ! -f "$DEST/SNGS-068.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-068 (6/57, PRIORIDADE) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-068]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-068.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-068.mp4"; echo "  -> $DEST/SNGS-068.mp4 ($(du -h "$DEST/SNGS-068.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-068" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 7/57 SNGS-072 (PRIORIDADE) ---
if [ ! -f "$DEST/SNGS-072.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-072 (7/57, PRIORIDADE) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-072]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-072.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-072.mp4"; echo "  -> $DEST/SNGS-072.mp4 ($(du -h "$DEST/SNGS-072.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-072" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 8/57 SNGS-075 (PRIORIDADE) ---
if [ ! -f "$DEST/SNGS-075.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-075 (8/57, PRIORIDADE) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-075]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-075.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-075.mp4"; echo "  -> $DEST/SNGS-075.mp4 ($(du -h "$DEST/SNGS-075.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-075" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 9/57 SNGS-077 (PRIORIDADE) ---
if [ ! -f "$DEST/SNGS-077.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-077 (9/57, PRIORIDADE) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-077]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-077.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-077.mp4"; echo "  -> $DEST/SNGS-077.mp4 ($(du -h "$DEST/SNGS-077.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-077" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 10/57 SNGS-060 (normal) ---
if [ ! -f "$DEST/SNGS-060.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-060 (10/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-060]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-060.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-060.mp4"; echo "  -> $DEST/SNGS-060.mp4 ($(du -h "$DEST/SNGS-060.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-060" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 11/57 SNGS-061 (normal) ---
if [ ! -f "$DEST/SNGS-061.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-061 (11/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-061]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-061.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-061.mp4"; echo "  -> $DEST/SNGS-061.mp4 ($(du -h "$DEST/SNGS-061.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-061" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 12/57 SNGS-062 (normal) ---
if [ ! -f "$DEST/SNGS-062.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-062 (12/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-062]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-062.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-062.mp4"; echo "  -> $DEST/SNGS-062.mp4 ($(du -h "$DEST/SNGS-062.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-062" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 13/57 SNGS-063 (normal) ---
if [ ! -f "$DEST/SNGS-063.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-063 (13/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-063]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-063.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-063.mp4"; echo "  -> $DEST/SNGS-063.mp4 ($(du -h "$DEST/SNGS-063.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-063" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 14/57 SNGS-065 (normal) ---
if [ ! -f "$DEST/SNGS-065.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-065 (14/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-065]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-065.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-065.mp4"; echo "  -> $DEST/SNGS-065.mp4 ($(du -h "$DEST/SNGS-065.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-065" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 15/57 SNGS-066 (normal) ---
if [ ! -f "$DEST/SNGS-066.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-066 (15/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-066]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-066.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-066.mp4"; echo "  -> $DEST/SNGS-066.mp4 ($(du -h "$DEST/SNGS-066.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-066" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 16/57 SNGS-067 (normal) ---
if [ ! -f "$DEST/SNGS-067.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-067 (16/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-067]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-067.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-067.mp4"; echo "  -> $DEST/SNGS-067.mp4 ($(du -h "$DEST/SNGS-067.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-067" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 17/57 SNGS-070 (normal) ---
if [ ! -f "$DEST/SNGS-070.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-070 (17/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-070]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-070.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-070.mp4"; echo "  -> $DEST/SNGS-070.mp4 ($(du -h "$DEST/SNGS-070.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-070" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 18/57 SNGS-071 (normal) ---
if [ ! -f "$DEST/SNGS-071.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-071 (18/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-071]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-071.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-071.mp4"; echo "  -> $DEST/SNGS-071.mp4 ($(du -h "$DEST/SNGS-071.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-071" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 19/57 SNGS-073 (normal) ---
if [ ! -f "$DEST/SNGS-073.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-073 (19/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-073]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-073.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-073.mp4"; echo "  -> $DEST/SNGS-073.mp4 ($(du -h "$DEST/SNGS-073.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-073" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 20/57 SNGS-074 (normal) ---
if [ ! -f "$DEST/SNGS-074.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-074 (20/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-074]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-074.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-074.mp4"; echo "  -> $DEST/SNGS-074.mp4 ($(du -h "$DEST/SNGS-074.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-074" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 21/57 SNGS-076 (normal) ---
if [ ! -f "$DEST/SNGS-076.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-076 (21/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-076]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-076.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-076.mp4"; echo "  -> $DEST/SNGS-076.mp4 ($(du -h "$DEST/SNGS-076.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-076" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 22/57 SNGS-097 (normal) ---
if [ ! -f "$DEST/SNGS-097.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-097 (22/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-097]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-097.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-097.mp4"; echo "  -> $DEST/SNGS-097.mp4 ($(du -h "$DEST/SNGS-097.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-097" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 23/57 SNGS-098 (normal) ---
if [ ! -f "$DEST/SNGS-098.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-098 (23/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-098]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-098.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-098.mp4"; echo "  -> $DEST/SNGS-098.mp4 ($(du -h "$DEST/SNGS-098.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-098" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 24/57 SNGS-099 (normal) ---
if [ ! -f "$DEST/SNGS-099.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-099 (24/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-099]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-099.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-099.mp4"; echo "  -> $DEST/SNGS-099.mp4 ($(du -h "$DEST/SNGS-099.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-099" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 25/57 SNGS-100 (normal) ---
if [ ! -f "$DEST/SNGS-100.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-100 (25/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-100]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-100.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-100.mp4"; echo "  -> $DEST/SNGS-100.mp4 ($(du -h "$DEST/SNGS-100.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-100" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 26/57 SNGS-101 (normal) ---
if [ ! -f "$DEST/SNGS-101.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-101 (26/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-101]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-101.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-101.mp4"; echo "  -> $DEST/SNGS-101.mp4 ($(du -h "$DEST/SNGS-101.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-101" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 27/57 SNGS-102 (normal) ---
if [ ! -f "$DEST/SNGS-102.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-102 (27/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-102]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-102.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-102.mp4"; echo "  -> $DEST/SNGS-102.mp4 ($(du -h "$DEST/SNGS-102.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-102" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 28/57 SNGS-103 (normal) ---
if [ ! -f "$DEST/SNGS-103.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-103 (28/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-103]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-103.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-103.mp4"; echo "  -> $DEST/SNGS-103.mp4 ($(du -h "$DEST/SNGS-103.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-103" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 29/57 SNGS-104 (normal) ---
if [ ! -f "$DEST/SNGS-104.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-104 (29/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-104]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-104.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-104.mp4"; echo "  -> $DEST/SNGS-104.mp4 ($(du -h "$DEST/SNGS-104.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-104" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 30/57 SNGS-105 (normal) ---
if [ ! -f "$DEST/SNGS-105.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-105 (30/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-105]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-105.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-105.mp4"; echo "  -> $DEST/SNGS-105.mp4 ($(du -h "$DEST/SNGS-105.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-105" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 31/57 SNGS-107 (normal) ---
if [ ! -f "$DEST/SNGS-107.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-107 (31/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-107]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-107.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-107.mp4"; echo "  -> $DEST/SNGS-107.mp4 ($(du -h "$DEST/SNGS-107.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-107" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 32/57 SNGS-108 (normal) ---
if [ ! -f "$DEST/SNGS-108.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-108 (32/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-108]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-108.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-108.mp4"; echo "  -> $DEST/SNGS-108.mp4 ($(du -h "$DEST/SNGS-108.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-108" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 33/57 SNGS-109 (normal) ---
if [ ! -f "$DEST/SNGS-109.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-109 (33/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-109]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-109.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-109.mp4"; echo "  -> $DEST/SNGS-109.mp4 ($(du -h "$DEST/SNGS-109.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-109" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 34/57 SNGS-110 (normal) ---
if [ ! -f "$DEST/SNGS-110.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-110 (34/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-110]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-110.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-110.mp4"; echo "  -> $DEST/SNGS-110.mp4 ($(du -h "$DEST/SNGS-110.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-110" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 35/57 SNGS-112 (normal) ---
if [ ! -f "$DEST/SNGS-112.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-112 (35/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-112]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-112.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-112.mp4"; echo "  -> $DEST/SNGS-112.mp4 ($(du -h "$DEST/SNGS-112.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-112" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 36/57 SNGS-114 (normal) ---
if [ ! -f "$DEST/SNGS-114.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-114 (36/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-114]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-114.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-114.mp4"; echo "  -> $DEST/SNGS-114.mp4 ($(du -h "$DEST/SNGS-114.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-114" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 37/57 SNGS-115 (normal) ---
if [ ! -f "$DEST/SNGS-115.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-115 (37/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-115]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-115.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-115.mp4"; echo "  -> $DEST/SNGS-115.mp4 ($(du -h "$DEST/SNGS-115.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-115" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 38/57 SNGS-151 (normal) ---
if [ ! -f "$DEST/SNGS-151.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-151 (38/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-151]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-151.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-151.mp4"; echo "  -> $DEST/SNGS-151.mp4 ($(du -h "$DEST/SNGS-151.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-151" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 39/57 SNGS-152 (normal) ---
if [ ! -f "$DEST/SNGS-152.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-152 (39/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-152]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-152.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-152.mp4"; echo "  -> $DEST/SNGS-152.mp4 ($(du -h "$DEST/SNGS-152.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-152" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 40/57 SNGS-153 (normal) ---
if [ ! -f "$DEST/SNGS-153.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-153 (40/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-153]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-153.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-153.mp4"; echo "  -> $DEST/SNGS-153.mp4 ($(du -h "$DEST/SNGS-153.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-153" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 41/57 SNGS-154 (normal) ---
if [ ! -f "$DEST/SNGS-154.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-154 (41/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-154]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-154.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-154.mp4"; echo "  -> $DEST/SNGS-154.mp4 ($(du -h "$DEST/SNGS-154.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-154" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 42/57 SNGS-155 (normal) ---
if [ ! -f "$DEST/SNGS-155.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-155 (42/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-155]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-155.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-155.mp4"; echo "  -> $DEST/SNGS-155.mp4 ($(du -h "$DEST/SNGS-155.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-155" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 43/57 SNGS-156 (normal) ---
if [ ! -f "$DEST/SNGS-156.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-156 (43/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-156]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-156.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-156.mp4"; echo "  -> $DEST/SNGS-156.mp4 ($(du -h "$DEST/SNGS-156.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-156" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 44/57 SNGS-157 (normal) ---
if [ ! -f "$DEST/SNGS-157.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-157 (44/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-157]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-157.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-157.mp4"; echo "  -> $DEST/SNGS-157.mp4 ($(du -h "$DEST/SNGS-157.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-157" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 45/57 SNGS-158 (normal) ---
if [ ! -f "$DEST/SNGS-158.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-158 (45/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-158]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-158.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-158.mp4"; echo "  -> $DEST/SNGS-158.mp4 ($(du -h "$DEST/SNGS-158.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-158" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 46/57 SNGS-159 (normal) ---
if [ ! -f "$DEST/SNGS-159.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-159 (46/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-159]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-159.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-159.mp4"; echo "  -> $DEST/SNGS-159.mp4 ($(du -h "$DEST/SNGS-159.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-159" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 47/57 SNGS-160 (normal) ---
if [ ! -f "$DEST/SNGS-160.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-160 (47/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-160]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-160.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-160.mp4"; echo "  -> $DEST/SNGS-160.mp4 ($(du -h "$DEST/SNGS-160.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-160" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 48/57 SNGS-161 (normal) ---
if [ ! -f "$DEST/SNGS-161.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-161 (48/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-161]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-161.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-161.mp4"; echo "  -> $DEST/SNGS-161.mp4 ($(du -h "$DEST/SNGS-161.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-161" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 49/57 SNGS-162 (normal) ---
if [ ! -f "$DEST/SNGS-162.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-162 (49/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-162]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-162.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-162.mp4"; echo "  -> $DEST/SNGS-162.mp4 ($(du -h "$DEST/SNGS-162.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-162" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 50/57 SNGS-163 (normal) ---
if [ ! -f "$DEST/SNGS-163.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-163 (50/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-163]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-163.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-163.mp4"; echo "  -> $DEST/SNGS-163.mp4 ($(du -h "$DEST/SNGS-163.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-163" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 51/57 SNGS-164 (normal) ---
if [ ! -f "$DEST/SNGS-164.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-164 (51/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-164]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-164.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-164.mp4"; echo "  -> $DEST/SNGS-164.mp4 ($(du -h "$DEST/SNGS-164.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-164" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 52/57 SNGS-165 (normal) ---
if [ ! -f "$DEST/SNGS-165.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-165 (52/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-165]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-165.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-165.mp4"; echo "  -> $DEST/SNGS-165.mp4 ($(du -h "$DEST/SNGS-165.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-165" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 53/57 SNGS-166 (normal) ---
if [ ! -f "$DEST/SNGS-166.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-166 (53/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-166]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-166.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-166.mp4"; echo "  -> $DEST/SNGS-166.mp4 ($(du -h "$DEST/SNGS-166.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-166" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 54/57 SNGS-167 (normal) ---
if [ ! -f "$DEST/SNGS-167.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-167 (54/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-167]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-167.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-167.mp4"; echo "  -> $DEST/SNGS-167.mp4 ($(du -h "$DEST/SNGS-167.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-167" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 55/57 SNGS-168 (normal) ---
if [ ! -f "$DEST/SNGS-168.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-168 (55/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-168]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-168.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-168.mp4"; echo "  -> $DEST/SNGS-168.mp4 ($(du -h "$DEST/SNGS-168.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-168" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 56/57 SNGS-169 (normal) ---
if [ ! -f "$DEST/SNGS-169.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-169 (56/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-169]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-169.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-169.mp4"; echo "  -> $DEST/SNGS-169.mp4 ($(du -h "$DEST/SNGS-169.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-169" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

# --- 57/57 SNGS-170 (normal) ---
if [ ! -f "$DEST/SNGS-170.mp4" ]; then
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc "0-9")
  if [ "$FREE" -lt 8 ]; then echo "[$(date)] ABORTADO: ${FREE}G livres" >> $SN/docs/render.log; exit 1; fi
  echo "[$(date)] SNGS-170 (57/57, normal) livre=${FREE}G" >> $SN/docs/render.log
  $TL -cn soccernet dataset.eval_set=train dataset.nvid=-1 dataset.nframes=-1 \
    'dataset.vids_dict.train=[SNGS-170]' eval_tracking=False \
    visualization.cfg.save_videos=True state.save_file=null > "$LOGD/SNGS-170.log" 2>&1
  M=$(ls -t $SN/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
  if [ -n "$M" ]; then mv "$M" "$DEST/SNGS-170.mp4"; echo "  -> $DEST/SNGS-170.mp4 ($(du -h "$DEST/SNGS-170.mp4" | cut -f1))" >> $SN/docs/render.log
  else echo "  FALHOU SNGS-170" >> $SN/docs/render.log; fi
  find $SN/outputs -name "*.pklz" -newermt "-2 hours" -delete 2>/dev/null
fi

echo "[$(date)] RENDERIZACAO CONCLUIDA" >> $SN/docs/render.log
