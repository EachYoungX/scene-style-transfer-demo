"""Standalone SD1.5 + ControlNet + IP-Adapter inference for the demo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline

from .a2_schedule import set_a2_schedule_step
from .models import ModelPaths


DEFAULT_PROMPT = "a coherent scene with painterly atmosphere, preserve content layout"
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, distorted, text, watermark, copied objects"


@dataclass(frozen=True)
class PipelineConfig:
    size: int = 512
    num_inference_steps: int = 30
    strength: float = 0.76
    guidance_scale: float = 5.8
    controlnet_scale: float = 0.72
    ip_adapter_scale: float = 0.9
    weight_name: str = "ip-adapter-plus_sd15.safetensors"
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT


@dataclass(frozen=True)
class GenerationResult:
    image: Image.Image
    elapsed_sec: float
    peak_allocated_gb: float
    seed: int
    reference_strength: float


class StyleTransferPipeline:
    """One lazily loaded, local model pipeline for one selected model root."""

    def __init__(self, model_root: str | Path, config: PipelineConfig | None = None) -> None:
        self.model_paths = ModelPaths.from_root(model_root)
        self.config = config or PipelineConfig()
        self._pipe: StableDiffusionControlNetImg2ImgPipeline | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def _ensure_loaded(self) -> StableDiffusionControlNetImg2ImgPipeline:
        if not torch.cuda.is_available():
            raise RuntimeError("需要 CUDA GPU 才能运行图像生成。")
        if self._pipe is None:
            controlnet = ControlNetModel.from_pretrained(
                self.model_paths.controlnet,
                torch_dtype=torch.float16,
                local_files_only=True,
                variant="fp16",
            )
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.model_paths.base_model,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
                variant="fp16",
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.load_ip_adapter(
                str(self.model_paths.ip_adapter),
                subfolder="models",
                weight_name=self.config.weight_name,
                local_files_only=True,
            )
            pipe.set_ip_adapter_scale(self.config.ip_adapter_scale)
            self._pipe = pipe.to("cuda")
        return self._pipe

    def generate(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        *,
        seed: int = 42,
        reference_strength: float = 0.6,
    ) -> GenerationResult:
        if not 0.2 <= reference_strength <= 1.0:
            raise ValueError("参考强度必须在 0.2 到 1.0 之间。")
        if seed < 0:
            raise ValueError("Seed 必须是非负整数。")

        pipe = self._ensure_loaded()
        content = fit_square_image(content_image, self.config.size)
        style = fit_square_crop(style_image, self.config.size)
        control = make_canny(content)
        embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=style,
            ip_adapter_image_embeds=None,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        pipe.scheduler.set_timesteps(self.config.num_inference_steps, device="cuda")
        set_a2_schedule_step(
            pipe,
            base_scale=self.config.ip_adapter_scale,
            reference_strength=reference_strength,
            step_index=0,
            num_steps=self.config.num_inference_steps,
        )

        def callback(pipe_ref: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
            set_a2_schedule_step(
                pipe_ref,
                base_scale=self.config.ip_adapter_scale,
                reference_strength=reference_strength,
                step_index=step_index,
                num_steps=self.config.num_inference_steps,
            )
            return callback_kwargs

        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        image = pipe(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            image=content,
            control_image=control,
            ip_adapter_image_embeds=embeds,
            strength=self.config.strength,
            guidance_scale=self.config.guidance_scale,
            controlnet_conditioning_scale=self.config.controlnet_scale,
            num_inference_steps=self.config.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            callback_on_step_end=callback,
        ).images[0]
        return GenerationResult(
            image=image,
            elapsed_sec=time.time() - start,
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
            seed=seed,
            reference_strength=reference_strength,
        )


def fit_square_image(image: Image.Image, size: int) -> Image.Image:
    source = image.convert("RGB").copy()
    source.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(source, ((size - source.width) // 2, (size - source.height) // 2))
    return canvas


def fit_square_crop(image: Image.Image, size: int) -> Image.Image:
    source = image.convert("RGB")
    side = min(source.width, source.height)
    left = (source.width - side) // 2
    top = (source.height - side) // 2
    return source.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.LANCZOS
    )


def make_canny(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    edges = cv2.Canny(np.asarray(image), low_threshold, high_threshold)
    return Image.fromarray(np.stack([edges, edges, edges], axis=-1))
