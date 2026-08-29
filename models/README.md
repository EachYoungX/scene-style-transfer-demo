# 模型目录说明

此目录只保存目录结构说明，不包含模型权重。你可以把模型复制到这里，也可以在 Demo 左侧“模型文件夹”中填写外部模型根目录。

推荐的项目内结构：

```text
models/
├── sd15/
│   ├── model_index.json
│   ├── unet/
│   ├── vae/
│   ├── text_encoder/
│   ├── tokenizer/
│   └── scheduler/
│       └── scheduler_config.json
├── controlnet_canny/
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors
└── ip_adapter_plus/
    └── models/
        └── ip-adapter-plus_sd15.safetensors
```

其中 `sd15/` 是 SD1.5 底座模型；`controlnet_canny/` 是 SD1.5 Canny ControlNet；`ip_adapter_plus/models/` 中是 IP-Adapter Plus 权重。`safety_checker/` 和模型目录中的其他标准文件可以一并保留，但 Demo 不会下载它们。

外部模型根目录也使用同样结构。例如填写 `/data/models` 后，程序会查找 `/data/models/sd15`、`/data/models/controlnet_canny` 和 `/data/models/ip_adapter_plus`。如果填写的路径本身就是 `sd15/`，程序也会自动尝试从其同级目录查找另外两个组件。
