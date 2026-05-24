# KubyterLab-IMG-12G

A JupyterLab Docker image with pre-installed Stable Diffusion models for AI image generation, inpainting, and image manipulation tasks. This 12GB variant includes eight models across multiple architectures, ready for immediate use.

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-sinanozel%2Fkubyterlab--img--12g-blue?logo=docker)](https://hub.docker.com/r/sinanozel/kubyterlab-img-12g)
[![Docker Pulls](https://img.shields.io/docker/pulls/sinanozel/kubyterlab-img-12g)](https://hub.docker.com/r/sinanozel/kubyterlab-img-12g)
[![Docker Image Size](https://img.shields.io/docker/image-size/sinanozel/kubyterlab-img-12g/26.04)](https://hub.docker.com/r/sinanozel/kubyterlab-img-12g)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Features

- **Pre-installed Models**: Eight models across SD 1.5 and SDXL architectures — no downloads required
- **GPU Acceleration**: CUDA-enabled with optimized fp16 inference
- **Ready-to-Use**: Start generating immediately from included sample notebooks
- **Multiple Tasks**: Text-to-image generation, inpainting, and image-to-image workflows

## 📦 Included Models

### SD 1.5 Architecture
*~2GB VRAM for weights. Recommended resolution: 512×512 to 768×768. Max safe resolution: 1024×1024.*

#### 1. CompVis Stable Diffusion v1.4
- **Source**: [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- **Use Case**: Text-to-image generation
- **Location**: `/jupyterlab/hf/hub/models--CompVis--stable-diffusion-v1-4`
- **Citation**:
  ```
  @InProceedings{Rombach_2022_CVPR,
      author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Björn},
      title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2022},
      pages     = {10684-10695}
  }
  ```

#### 2. Fluently v3 Inpainting
- **Source**: [fluently/Fluently-v3-inpainting](https://huggingface.co/fluently/Fluently-v3-inpainting)
- **Use Case**: Image inpainting and editing
- **Location**: `/jupyterlab/hf/hub/models--fluently--Fluently-v3-inpainting`
- **License**: CreativeML Open RAIL-M

#### 3. Fluently v4 Inpainting
- **Source**: [fluently/Fluently-v4-inpainting](https://huggingface.co/fluently/Fluently-v4-inpainting)
- **Use Case**: Image inpainting and editing (improved v3)
- **Location**: `/jupyterlab/hf/hub/models--fluently--Fluently-v4-inpainting`
- **License**: fluently-license

#### 4. DreamShaper 8 Inpainting
- **Source**: [Lykon/dreamshaper-8-inpainting](https://huggingface.co/Lykon/dreamshaper-8-inpainting)
- **Use Case**: Artistic and stylized inpainting
- **Location**: `/jupyterlab/hf/hub/models--Lykon--dreamshaper-8-inpainting`

#### 5. Realistic Vision v6
- **Source**: [SG161222/Realistic_Vision_V6.0_B1_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE)
- **Use Case**: Photorealistic image generation
- **Location**: `/jupyterlab/hf/hub/models--SG161222--Realistic_Vision_V6.0_B1_noVAE`

### SDXL Architecture
*~7–8GB VRAM for weights. Recommended resolution: 1024×1024. Always load with `torch_dtype=torch.float16` and `variant="fp16"`.*

#### 6. DreamShaper XL v2 Turbo
- **Source**: [Lykon/dreamshaper-xl-v2-turbo](https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo)
- **Use Case**: High-quality artistic generation at 1024px
- **Location**: `/jupyterlab/hf/hub/models--Lykon--dreamshaper-xl-v2-turbo`

#### 7. Stable Diffusion XL Base 1.0
- **Source**: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **Use Case**: High-quality text-to-image generation at 1024px
- **Location**: `/jupyterlab/hf/hub/models--stabilityai--stable-diffusion-xl-base-1.0`

## 🖥️ VRAM and Resolution Guidelines

Always resize source images before passing them to the pipeline. The pipeline is not designed to handle arbitrary input sizes gracefully.

| Architecture | Model VRAM | Init Image | Output | Notes |
|---|---|---|---|---|
| SD 1.5 | ~2GB | up to 1024×1024 | up to 1024×1024 | Sweet spot is 512–768px. Quality degrades above 768px. |
| SDXL | ~8GB | 1024×1024 | 1024×1024 | Native training resolution. Stay here. |

**General rules:**
- Init image resolution should match output resolution. Mismatching causes internal rescaling and wastes both VRAM and quality.
- Always resize source images *before* passing to the pipeline, not inside it.
- Use batch size 1 at all times on 12GB VRAM.
- If you get a CUDA OOM error, halve the resolution before trying anything else.

```python
from PIL import Image

def prepare_image(path, size=1024):
    img = Image.open(path)
    img.thumbnail((size, size), Image.LANCZOS)
    return img
```

## 🛠️ Usage

### Quick Start

```bash
# With your own notebooks
docker run --gpus all -p 8888:8888 \
  -v $(pwd)/notebooks:/jupyterlab/notebooks \
  sinanozel/kubyterlab-img-12g:26.04

# Ephemeral — explore example notebooks only
docker run --gpus all -p 8888:8888 sinanozel/kubyterlab-img-12g:26.04
```

> ⚠️ **Important**: Without mounting a volume, any changes will be lost when the container stops.

### With Docker Compose

```yaml
version: '3.8'
services:
  jupyter:
    image: sinanozel/kubyterlab-img-12g:26.04
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/jupyterlab/notebooks
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📊 Sample Code

### SD 1.5 — Text-to-Image Generation (CompVis SD v1.4)

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

image = pipe(
    "a fantasy landscape with mountains and rivers",
    width=512,
    height=512
).images[0]
image.save("generated_landscape.png")
```

### SD 1.5 — Photorealistic Generation (Realistic Vision v6)

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

image = pipe(
    "portrait of a person in natural light, photorealistic, 8k",
    negative_prompt="low quality, blurry, cartoon, painting",
    width=768,
    height=768,
    num_inference_steps=30,
).images[0]
image.save("portrait.png")
```

### SD 1.5 — Inpainting (Fluently v3 or v4)

```python
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

# Use v3 or v4 — same interface
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "fluently/Fluently-v4-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

base_image = Image.open("your_image.png").resize((512, 512))
mask = Image.open("your_mask.png").resize((512, 512))  # white = area to inpaint

result = pipe(
    prompt="a beautiful garden with flowers",
    negative_prompt="low quality, blurry",
    image=base_image,
    mask_image=mask,
    num_inference_steps=30,
).images[0]
result.save("inpainted.png")
```

### SD 1.5 — Artistic Inpainting (DreamShaper 8)

```python
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "Lykon/dreamshaper-8-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

base_image = Image.open("your_image.png").resize((768, 768))
mask = Image.open("your_mask.png").resize((768, 768))

result = pipe(
    prompt="oil painting of a mountain landscape, dramatic lighting, artstation",
    negative_prompt="low quality, blurry, photographic",
    image=base_image,
    mask_image=mask,
    num_inference_steps=30,
).images[0]
result.save("artistic_inpaint.png")
```

### SDXL — Text-to-Image Generation (SDXL Base 1.0)

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

image = pipe(
    "a fantasy landscape with mountains and rivers, highly detailed, 4k",
    negative_prompt="low quality, blurry",
    width=1024,
    height=1024,
    num_inference_steps=30,
).images[0]
image.save("sdxl_landscape.png")
```

### SDXL — Artistic Generation (DreamShaper XL)

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "Lykon/dreamshaper-xl-v2-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

image = pipe(
    "cinematic portrait, dramatic lighting, ultra detailed",
    negative_prompt="low quality, blurry, bad anatomy",
    width=1024,
    height=1024,
    num_inference_steps=8,   # turbo model — fewer steps needed
    guidance_scale=2.0,      # turbo model — lower CFG
).images[0]
image.save("dreamshaper_xl.png")
```

## 📝 Included Notebooks

1. **`stable-diffusion-v1-4-demo.ipynb`**: Text-to-image generation with CompVis SD v1.4
2. **`fluently-v3-inpainting-demo.ipynb`**: Inpainting workflows with Fluently v3

## 🔧 System Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: 12GB recommended (SD 1.5 models work on 6GB+; SDXL requires 10GB+)
- **Docker**: Docker with GPU support (nvidia-container-toolkit)

## 🏗️ Build Information

- **Base Image**: `sinanozel/kubyterlab-img:26.04`
- **CUDA**: 12.9.2
- **Python**: 3.12
- **Key Dependencies**: diffusers, transformers, accelerate, safetensors, xformers, torch 2.8.0

## 📄 License

This container is licensed under MIT. Individual model licenses apply:

| Model | License |
|---|---|
| CompVis Stable Diffusion v1.4 | CreativeML Open RAIL-M |
| Fluently v3 Inpainting | CreativeML Open RAIL-M |
| Fluently v4 Inpainting | fluently-license |
| DreamShaper 8 Inpainting | see model card |
| Realistic Vision v6 | see model card |
| DreamShaper XL v2 Turbo | see model card |
| Stable Diffusion XL Base 1.0 | CreativeML Open RAIL++-M |

Please review individual model licenses before commercial use.

## 🤝 Contributing

This image is part of the [jupyterlab-on-kubernetes](https://github.com/sinan-ozel/jupyterlab-on-kubernetes) project. Contributions welcome!

## ⚠️ Responsible AI Usage

These models are powerful tools for creative work. Please use them responsibly:
- Respect copyright and intellectual property
- Avoid generating harmful or inappropriate content
- Follow the individual model license terms
- Consider the ethical implications of AI-generated content
- Please make it clear to others that you used AI. People can tell if they look closely anyway, so even if you misrepresent your own contribution, it will eventually be obvious.

---

For more information, visit the [main project repository](https://github.com/sinan-ozel/jupyterlab-on-kubernetes).