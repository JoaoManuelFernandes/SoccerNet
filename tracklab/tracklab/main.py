"""
tracklab/tracklab/main.py
Ponto de entrada do TrackLab + instrumentação de memória
"""

import os
import warnings
import logging
# --- 3rd-party -------------------------------------------------------------
import hydra
import rich.logging
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# --- TrackLab --------------------------------------------------------------
from tracklab.utils import progress, wandb          # noqa: E402
from tracklab.datastruct import TrackerState        # noqa: E402
from tracklab.pipeline import Pipeline              # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# MEM DEBUG ─ imprime VRAM (GPU) e RAM (CPU) em cada etapa
# ═══════════════════════════════════════════════════════════════════════════
import psutil                                     # noqa: E402

def mem(tag: str):
    """Print VRAM e RAM usadas com um rótulo 'tag'."""
    try:
        vram = torch.cuda.memory_allocated() / 1e6  # MB
    except Exception:
        vram = 0
    ram = psutil.Process(os.getpid()).memory_info().rss / 1e6
    print(f"[MEM] {tag:<28} | VRAM {vram:6.0f} MB | RAM {ram:6.0f} MB")

# ═══════════════════════════════════════════════════════════════════════════

os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@hydra.main(version_base=None,
            config_path="pkg://tracklab.configs",
            config_name="config")
def main(cfg):

    # --- diagnóstico opcional: imprime cfg final ---------------------------
    # print("\n=== CONFIG FINAL ===")
    # print(OmegaConf.to_yaml(cfg))
    # print("============================================================\n")

    #mem("start main")

    device = init_environment(cfg)
    #mem("after init_environment")

    # DATASET & EVALUATOR ---------------------------------------------------
    tracking_dataset = instantiate(cfg.dataset)
    #mem("after dataset")

    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)
    #mem("after evaluator")

    # MODULES ---------------------------------------------------------------
    modules = []
    if cfg.pipeline is not None:
        for name in cfg.pipeline:
            log.info(f"→ instantiating module '{name}'")
            module_cfg = cfg.modules[name]
            inst_module = instantiate(module_cfg,
                                      device=device,
                                      tracking_dataset=tracking_dataset)
            modules.append(inst_module)
            #mem(f"after module {name}")

    pipeline = Pipeline(models=modules)
    #mem("after Pipeline")

    # TRAIN (se activado) ---------------------------------------------------
    for module in modules:
        if getattr(module, "training_enabled", False):
            module.train()
            #mem(f"after training {module.__class__.__name__}")

    # TRACKING --------------------------------------------------------------
    if cfg.test_tracking:
        log.info(f"Starting tracking on '{cfg.dataset.eval_set}' set.")
        tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
        tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
        tracking_engine = instantiate(cfg.engine,
                                      modules=pipeline,
                                      tracker_state=tracker_state)

        #mem("before track_dataset")
        tracking_engine.track_dataset()
        #mem("after track_dataset")

        evaluate(cfg, evaluator, tracker_state)

        if tracker_state.save_file is not None:
            log.info(f"Saved state at : {tracker_state.save_file.resolve()}")

    close_environment()
    #mem("end main")
    return 0


# ────────────────────────────────────────────────────────────────────────────
def set_sharing_strategy():
    # Evita deadlocks de multiprocessamento PyTorch em Linux
    torch.multiprocessing.set_sharing_strategy("file_system")


def init_environment(cfg):
    progress.use_rich = cfg.use_rich
    set_sharing_strategy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: '{device}'.")

    # logger / wandb --------------------------------------------------------
    wandb.init(cfg)

    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))

    if cfg.use_rich:
        # silencia StreamHandler padrão para evitar duplicação com Rich
        for h in log.root.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(logging.ERROR)
        log.root.addHandler(rich.logging.RichHandler(level=logging.INFO))
    else:
        for h in log.root.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(logging.INFO)

    return device


def close_environment():
    wandb.finish()


def evaluate(cfg, evaluator, tracker_state):
    if cfg.get("eval_tracking", True) and cfg.dataset.nframes == -1:
        log.info("Starting evaluation.")
        evaluator.run(tracker_state)
    elif cfg.get("eval_tracking", True) is False:
        log.warning("Skipping evaluation because 'eval_tracking' is False.")
    else:
        log.warning("Skipping evaluation because only part of the video was tracked.")


if __name__ == "__main__":
    main()
