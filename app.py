"""Chinese single-page Gradio wrapper for the standalone scene demo."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

from config import (
    DEFAULT_MODEL_ROOT,
    DEFAULT_REFERENCE_STRENGTH,
    DEFAULT_SEED,
    OUTPUT_DIR,
    create_pipeline,
)


PIPELINE = None
PIPELINE_ROOT: Path | None = None


def get_pipeline(model_root: str | None):
    global PIPELINE, PIPELINE_ROOT
    root = Path(model_root).expanduser().resolve() if model_root and model_root.strip() else DEFAULT_MODEL_ROOT
    if PIPELINE is None or PIPELINE_ROOT != root:
        PIPELINE = create_pipeline(root)
        PIPELINE_ROOT = root
    return PIPELINE


def generate_result(
    content_image: Image.Image | None,
    style_image: Image.Image | None,
    model_root: str | None,
    reference_strength: float,
    seed: float,
):
    """Run one generation and persist a downloadable PNG."""

    if content_image is None or style_image is None:
        return None, None, "请先上传原图片和参考图片。"

    try:
        seed_value = int(seed)
        strength_value = float(reference_strength)
        result = get_pipeline(model_root).generate(
            content_image,
            style_image,
            seed=seed_value,
            reference_strength=strength_value,
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = OUTPUT_DIR / f"scene_style_transfer_{stamp}_seed{seed_value}.png"
        result.image.save(output_path)
        vram = (
            f"峰值显存：{result.peak_allocated_gb:.2f} GB"
            if result.peak_allocated_gb is not None
            else "峰值显存：不可用"
        )
        status = (
            f"已生成 · Seed：`{seed_value}` · 参考强度：`{strength_value:.1f}`  \n"
            f"耗时：`{result.elapsed_sec:.1f} 秒` · {vram}"
        )
        return result.image, str(output_path), status
    except Exception as exc:
        return None, None, f"生成失败：`{exc}`"


def reset_advanced():
    return DEFAULT_REFERENCE_STRENGTH, DEFAULT_SEED


CSS = """
body { background: #ffffff; color: #111111; }
.gradio-container { max-width: 980px !important; margin: 0 auto; }
.input-panel, .result-panel { border: 1px solid #d8d8d8; border-radius: 6px; padding: 12px; }
button.primary { background: #111111 !important; border-color: #111111 !important; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Scene Style Transfer", css=CSS) as demo:
        gr.Markdown(
            "# 场景风格迁移\n"
            "使用参考图片进行场景风格迁移，同时尽量保持原图片的空间结构。"
        )
        with gr.Row():
            with gr.Column(elem_classes="input-panel"):
                gr.Markdown("### 输入与参数")
                content = gr.Image(
                    type="pil", sources=["upload"], label="原图片（Content Image）", height=180
                )
                style = gr.Image(
                    type="pil", sources=["upload"], label="参考图片（Style Reference）", height=180
                )
                model_root = gr.Textbox(
                    value=str(DEFAULT_MODEL_ROOT),
                    label="模型文件夹",
                    info="可填写外部模型根目录；默认使用项目内 models/。",
                )
                gr.Markdown("底座、ControlNet 和 IP-Adapter 的目录结构见 `models/README.md`。")
                with gr.Accordion("高级参数", open=False):
                    reference_strength = gr.Slider(
                        minimum=0.2,
                        maximum=1.0,
                        value=DEFAULT_REFERENCE_STRENGTH,
                        step=0.1,
                        label="参考强度",
                        info="推荐 operating point：0.6",
                    )
                    seed = gr.Number(value=DEFAULT_SEED, precision=0, label="Seed（随机种子）")
                    reset = gr.Button("恢复推荐参数")
                generate = gr.Button("生成图片", variant="primary")

            with gr.Column(elem_classes="result-panel"):
                gr.Markdown("### 生成结果")
                result = gr.Image(type="pil", label="结果图片", format="png", height=560)
                download = gr.File(label="下载已保存结果")
                status = gr.Markdown("上传原图片和参考图片后，点击“生成图片”。")

        generate.click(
            fn=generate_result,
            inputs=[content, style, model_root, reference_strength, seed],
            outputs=[result, download, status],
        )
        reset.click(fn=reset_advanced, outputs=[reference_strength, seed])

    return demo


if __name__ == "__main__":
    build_demo().launch()
