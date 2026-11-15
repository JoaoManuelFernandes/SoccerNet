#!/bin/bash
# Reconstroi o ambiente Python (venv_tl) apos a perda do miniconda.
# Corre DEPOIS de torch+torchvision estarem instalados.
P=/home/joao/soccernet/venv_tl/bin/python
set -x
$P -m pip install --no-cache-dir \
  "numpy==1.26.4" "pandas" "scikit-learn" "matplotlib<3.9" "Pillow" "scipy" \
  "opencv-python" "hydra-core>=1.3" "omegaconf" "rich" "psutil" "pyyaml" "tqdm" \
  "tabulate" "yacs" "filterpy" "gdown" "requests" "protobuf" "six" \
  "lightning" "pytorch_lightning" "torchmetrics" \
  "timm" "ultralytics" "yt-dlp" "soccernet" "sn-trackeval" \
  "kornia" "lsq-ellipse" "scikit-image" "seaborn" "imageio" "imageio-ffmpeg"
$P -m pip install --no-cache-dir "transformers>=4.49" "accelerate" "bitsandbytes" "qwen-vl-utils" "safetensors" "sentencepiece" "einops"
$P -m pip install --no-cache-dir "lap"
$P -m pip install --no-cache-dir --no-deps "git+https://github.com/VlSomers/prtreid"
$P -m pip install --no-cache-dir --no-deps "git+https://github.com/VlSomers/bpbreid"
$P -m pip install --no-deps -e /home/joao/soccernet/tracklab
$P -m pip install --no-deps -e /home/joao/soccernet/sn-gamestate/plugins/calibration
$P -m pip install --no-deps -e /home/joao/soccernet/sn-gamestate
set +x
echo "=== VERIFICACAO ==="
$P - <<'PY'
mods = ["torch","torchvision","numpy","pandas","cv2","hydra","ultralytics","transformers",
        "accelerate","bitsandbytes","qwen_vl_utils","lightning","timm","sklearn","matplotlib",
        "filterpy","lap","kornia","prtreid","torchreid","trackeval","tracklab","sn_gamestate",
        "nbjw_calib","sn_calibration_baseline","SoccerNet"]
import importlib
for m in mods:
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except Exception as e:
        print(f"  FALTA {m}: {type(e).__name__}: {str(e)[:90]}")
import torch; print("CUDA:", torch.cuda.is_available(), torch.version.cuda)
PY
df -h / | tail -1
