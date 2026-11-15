#!/bin/bash
# Só o passo 6 de refaz_148.sh (o override precisava de ++ porque o vids_dict de valid não existe na config).
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
SN=/home/joao/soccernet/sn-gamestate; TL=/home/joao/soccernet/venv_tl/bin/tracklab; OUT=/home/joao/soccernet/baseline_out; A=SNGS-050
(cd "$SN" && $TL -cn soccernet num_cores=3 dataset.eval_set=valid dataset.nvid=-1 dataset.nframes=-1 \
    "++dataset.vids_dict.valid=[$A]" eval_tracking=False "pipeline=[]" \
    state.load_file=/home/joao/soccernet/nosso_out/states/$A.pklz state.save_file=null \
    visualization.cfg.save_videos=True > "$OUT/logs_render/$A.nosso.log" 2>&1)
M=$(ls -t "$SN"/outputs/sn-gamestate/*/*/visualization/videos/*.mp4 2>/dev/null | head -1)
[ -n "$M" ] && mv "$M" "$OUT/videos/${A}_nosso.mp4" && echo "FIM $A" >> "$OUT/refaz_148.log" || echo "render $A FALHOU" >> "$OUT/refaz_148.log"
