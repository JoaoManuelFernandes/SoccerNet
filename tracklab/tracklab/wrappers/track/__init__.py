import warnings

try:
    from .strong_sort_api import StrongSORT
except Exception as e:
    warnings.warn(f"[tracklab] StrongSORT não disponível: {e}")

try:
    from .bot_sort_api import BotSORT
except Exception as e:
    warnings.warn(f"[tracklab] BotSORT não disponível: {e}")

try:
    from .byte_track_api import ByteTrack
except Exception as e:
    warnings.warn(f"[tracklab] ByteTrack não disponível: {e}")

try:
    from .oc_sort_api import OCSORT
except Exception as e:
    warnings.warn(f"[tracklab] OCSORT não disponível: {e}")

try:
    from .deep_oc_sort_api import DeepOCSORT
except Exception as e:
    warnings.warn(f"[tracklab] DeepOCSORT não disponível: {e}")

try:
    from .bpbreid_strong_sort_api import BPBReIDStrongSORT
except Exception as e:
    warnings.warn(f"[tracklab] BPBReIDStrongSORT não disponível: {e}")
