"""Self-contained inference pieces used by the standalone demo."""

from .models import ModelPaths
from .pipeline import GenerationResult, PipelineConfig, StyleTransferPipeline

__all__ = ["GenerationResult", "ModelPaths", "PipelineConfig", "StyleTransferPipeline"]
