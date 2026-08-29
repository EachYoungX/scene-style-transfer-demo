"""Minimal copy of the frozen A2 high-resolution injection schedule."""

from __future__ import annotations

from typing import Any


def set_a2_schedule_step(
    pipe: Any,
    *,
    base_scale: float,
    reference_strength: float,
    step_index: int,
    num_steps: int,
) -> None:
    """Enable A2 on up blocks and keep down/mid IP branches disabled.

    The demo exposes a uniform reference strength. Multiplying the processor
    scale by it is equivalent to the research implementation's uniform spatial
    gate, while avoiding the diagnostics-only processor copy.
    """

    del step_index, num_steps  # A2 has no timestep curve; it is fixed over time.
    for name, processor in pipe.unet.attn_processors.items():
        if not hasattr(processor, "scale"):
            continue
        enabled = name.startswith("up_blocks.")
        value = float(base_scale * reference_strength) if enabled else 0.0
        processor.scale = [value for _ in processor.scale] if isinstance(processor.scale, list) else value
