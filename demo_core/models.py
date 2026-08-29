"""Model-directory discovery for the standalone demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_MODEL_MARKERS = (
    Path("model_index.json"),
    Path("unet/config.json"),
    Path("vae/config.json"),
    Path("text_encoder/config.json"),
    Path("tokenizer"),
    Path("scheduler/scheduler_config.json"),
)


@dataclass(frozen=True)
class ModelPaths:
    """Resolved paths for the three local components used by the pipeline."""

    base_model: Path
    controlnet: Path
    ip_adapter: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ModelPaths":
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"模型文件夹不存在：{root}")

        base_model = _first_existing(
            root / "sd15",
            root / "stable-diffusion-v1-5",
            root,
        )
        if base_model is None:
            raise FileNotFoundError(
                "未找到 SD1.5 底座模型。请将模型目录设置为包含 model_index.json、"
                "unet/、vae/、text_encoder/、tokenizer/ 和 scheduler/ 的文件夹，"
                "或设置为包含 sd15/ 子目录的模型根目录。"
            )

        sibling_root = base_model.parent
        controlnet = _first_dir(
            root / "controlnet_canny",
            root / "controlnet",
            root / "control_v11p_sd15_canny",
            sibling_root / "controlnet_canny",
        )
        ip_adapter = _first_dir(
            root / "ip_adapter_plus",
            root / "ip_adapter",
            root / "ip-adapter-plus",
            sibling_root / "ip_adapter_plus",
        )
        missing = []
        if controlnet is None:
            missing.append("controlnet_canny/")
        if ip_adapter is None:
            missing.append("ip_adapter_plus/")
        if missing:
            raise FileNotFoundError(
                f"模型根目录 {root} 缺少：{', '.join(missing)}。"
                "请参考 models/README.md 的目录结构。"
            )

        paths = cls(base_model=base_model, controlnet=controlnet, ip_adapter=ip_adapter)
        paths.validate()
        return paths

    def validate(self) -> None:
        missing = [str(path) for path in BASE_MODEL_MARKERS if not (self.base_model / path).exists()]
        if not (self.controlnet / "config.json").exists():
            missing.append(str(self.controlnet / "config.json"))
        if not _contains_weight(self.controlnet):
            missing.append(f"{self.controlnet}/diffusion_pytorch_model(.fp16).safetensors")
        if not (self.ip_adapter / "models").is_dir():
            missing.append(str(self.ip_adapter / "models"))
        if not _contains_weight(self.ip_adapter / "models", prefix="ip-adapter"):
            missing.append(f"{self.ip_adapter}/models/ip-adapter-plus_sd15.safetensors")
        if missing:
            raise FileNotFoundError("模型目录缺少必要文件：\n- " + "\n- ".join(missing))


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "model_index.json").exists():
            return candidate
    return None


def _first_dir(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _contains_weight(directory: Path, prefix: str | None = None) -> bool:
    if not directory.is_dir():
        return False
    patterns = ("*.safetensors", "*.bin", "*.ckpt")
    return any(
        path.is_file() and (prefix is None or path.name.lower().startswith(prefix.lower()))
        for pattern in patterns
        for path in directory.glob(pattern)
    )
