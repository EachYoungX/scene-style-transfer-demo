"""Local configuration for the standalone research demo."""

from __future__ import annotations

import os
from pathlib import Path

from demo_core import StyleTransferPipeline


DEMO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = Path(
    os.environ.get("SCENE_STYLE_TRANSFER_MODEL_ROOT", str(DEMO_ROOT / "models"))
).expanduser().resolve()
OUTPUT_DIR = DEMO_ROOT / "outputs"
DEFAULT_REFERENCE_STRENGTH = 0.6
DEFAULT_SEED = 42


def create_pipeline(model_root: str | Path | None = None) -> StyleTransferPipeline:
    """Create a lazy pipeline from the selected local model root."""

    return StyleTransferPipeline(model_root or DEFAULT_MODEL_ROOT)
