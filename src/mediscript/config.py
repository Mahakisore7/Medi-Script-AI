"""Central configuration: filesystem layout and runtime settings.

Every path in the code base is derived from ``PROJECT_ROOT`` so scripts work
regardless of the current working directory. Secrets are never stored in the
repository; they are read from the environment (see ``.env.example``).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("MEDISCRIPT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
LEXICON_DIR = Path(os.environ.get("MEDISCRIPT_LEXICON_DIR", DATA_DIR / "lexicon"))
MODELS_DIR = Path(os.environ.get("MEDISCRIPT_MODELS_DIR", PROJECT_ROOT / "models"))
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

YOLO_WEIGHTS = Path(
    os.environ.get("MEDISCRIPT_YOLO_WEIGHTS", MODELS_DIR / "yolo" / "yolo_real_world_best.pt")
)

# TrOCR checkpoints are tried newest-first; falls back to the public base model.
TROCR_CHECKPOINTS = [
    MODELS_DIR / "trocr" / "final_trocr_large_model_v4_dosage",
    MODELS_DIR / "trocr" / "final_trocr_large_model_v3",
    MODELS_DIR / "trocr" / "final_trocr_large_model_v2",
    MODELS_DIR / "trocr" / "final_trocr_large_model",
]
TROCR_BASE = "microsoft/trocr-large-handwritten"

GEMINI_MODEL = os.environ.get("MEDISCRIPT_GEMINI_MODEL", "gemini-2.5-flash")
# Seconds to wait before each LLM call (free-tier rate limit); set 0 on paid tiers.
LLM_DELAY_S = float(os.environ.get("MEDISCRIPT_LLM_DELAY", "4"))


def gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
