#!/bin/bash
# v2 — corrigido data_dir para path absoluto
source /home/joao/miniconda3/etc/profile.d/conda.sh
conda activate tracklab
cd /home/joao/soccernet/sn-gamestate
tracklab -cn soccernet dataset.nframes=20 dataset.nvid=1 2>&1 | tee /tmp/pipeline_run.log
