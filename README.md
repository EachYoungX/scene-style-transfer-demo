# Scene Style Transfer Demo

这是一个基于项目 Scene Style Transfer 的展示 Demo。只包含运行展示所需的精简推理适配代码，需按要求下载或选择指定底座模型权重。

## 功能

- 上传一张原图片和一张参考图片；
- 默认使用推荐 operating point：`参考强度=0.6`、`Seed=42`；
- 高级参数只开放参考强度和 Seed；
- 左侧选择模型文件夹，右侧显示结果；
- 一次生成一张结果图，并保存到 `outputs/`；
- 显示运行时间和峰值显存。

## 安装与运行

### Conda 环境

Demo 使用 Python 3.10、PyTorch 2.5.1、CUDA 12.4 和 Gradio 5。可以直接使用仓库里的 [environment.yml](environment.yml)：

```bash
cd ~/Workspace/Development/Projects/PythonProjects/scene-style-transfer-demo
conda env create -f environment.yml
conda activate sst_demo_env
```

如果环境已经存在，更新配置：

```bash
conda env update -n sst_demo_env -f environment.yml --prune
conda activate sst_demo_env
```

也可以手动创建同样的环境：

```bash
conda create -n sst_demo_env python=3.10 pip -y
conda activate sst_demo_env
conda install pytorch=2.5.1 torchvision=0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
python -m pip install -r requirements.txt
```

### uv 环境

```bash
uv venv sst_demo_env --python 3.10
uv pip install --python sst_demo_env/bin/python -r requirements.txt
source sst_demo_env/bin/activate
```

### 启动 Demo

```bash
conda activate sst_demo_env  # 或 source sst_demo_env/bin/activate
python app.py
```

启动后，在浏览器打开终端输出的本地地址。Demo 需要 CUDA GPU；首次生成时会加载本地模型，之后同一进程内会复用已加载的模型。

## 模型下载

模型权重不包含在仓库中。先安装依赖并激活环境；如果 Hugging Face 需要授权，先执行：

```bash
hf auth login
```

以下命令在 Demo 根目录执行，下载到项目内的 `models/`：

```bash
mkdir -p models/sd15 models/controlnet_canny models/ip_adapter_plus

hf download runwayml/stable-diffusion-v1-5 \
  --local-dir models/sd15 \
  --include model_index.json feature_extractor/* scheduler/* \
  text_encoder/config.json text_encoder/model.fp16.safetensors \
  tokenizer/* unet/config.json unet/diffusion_pytorch_model.fp16.safetensors \
  vae/config.json vae/diffusion_pytorch_model.fp16.safetensors

hf download lllyasviel/control_v11p_sd15_canny \
  --local-dir models/controlnet_canny \
  --include config.json "*.fp16.safetensors"

hf download h94/IP-Adapter \
  --local-dir models/ip_adapter_plus \
  --include models/image_encoder/config.json \
  models/image_encoder/model.safetensors \
  models/ip-adapter-plus_sd15.safetensors
```

下载完成后，目录应为：

```text
models/
├── sd15/
├── controlnet_canny/
└── ip_adapter_plus/models/
```

如果不想复制模型，可以在界面左侧“模型文件夹”填写已有的外部模型根目录。完整文件清单见 [models/README.md](models/README.md)。
