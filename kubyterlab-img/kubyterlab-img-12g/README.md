# KubyterLab-IMG-12G

A JupyterLab Docker image with pre-installed Stable Diffusion models for AI image generation and inpainting tasks. This 12GB variant includes two powerful models ready for immediate use.

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-sinanozel%2Fkubyterlab--img--12g-blue?logo=docker)](https://hub.docker.com/r/sinanozel/kubyterlab-img-12g)
[![Docker Pulls](https://img.shields.io/docker/pulls/sinanozel/kubyterlab-img-12g)](https://hub.docker.com/r/sinanozel/kubyterlab-img-12g)
[![Docker Image Size](https://img.shields.io/docker/image-size/sinanozel/kubyterlab-img-12g/25.11)](https://hub.docker.com/r/sinanozel/kubyterlab-img-12g)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Features

- **Pre-installed Models**: CompVis Stable Diffusion v1.4 and Fluently v3 Inpainting
- **GPU Acceleration**: CUDA-enabled with optimized dependencies
- **Ready-to-Use**: No model download required - start generating immediately
- **Sample Notebooks**: Included examples for both text-to-image and inpainting workflows

## 📦 Included Models

### 1. CompVis Stable Diffusion v1.4
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

### 2. Fluently v3 Inpainting
- **Source**: [fluently/Fluently-v3-inpainting](https://huggingface.co/fluently/Fluently-v3-inpainting)
- **Use Case**: Image inpainting and editing
- **Location**: `/jupyterlab/hf/hub/models--fluently--Fluently-v3-inpainting`
- **License**: CreativeML Open RAIL-M

## 🛠️ Usage

### Quick Start

#### With Your Own Notebooks
```bash
docker run --gpus all -p 8888:8888 -v $(pwd):/jupyterlab/notebooks sinanozel/kubyterlab-img-12g:25.11

# With persistent JupyterLab configuration (Dark Mode, etc.)
docker run --gpus all -p 8888:8888 \
  -v $(pwd)/notebooks:/jupyterlab/notebooks \
  -v $(pwd)/data/jupyter:/home/jovyan/.jupyter \
  -v $(pwd)/data/jupyterlab:/home/jovyan/.local/share/jupyter/lab \
  sinanozel/kubyterlab-img-12g:25.11
```

#### Try the Example Notebooks (Ephemeral)
If you want to explore the included example notebooks without mounting your own directory:
```bash
docker run --gpus all -p 8888:8888 sinanozel/kubyterlab-img-12g:25.11
```
> ⚠️ **Important**: Without mounting a volume, any changes you make will be **ephemeral** and lost when the container stops. The included example notebooks (`stable-diffusion-v1-4-demo.ipynb` and `fluently-v3-inpainting-demo.ipynb`) are perfect for exploring the capabilities, but save your work to a mounted volume for persistence.

### With Docker Compose
```yaml
version: '3.8'
services:
  jupyter:
    image: sinanozel/kubyterlab-img-12g:25.11
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/jupyterlab/notebooks
      # For persistent JupyterLab settings (Dark Mode, extensions, etc.)
      - ./data/jupyter:/home/jovyan/.jupyter
      - ./data/jupyterlab:/home/jovyan/.local/share/jupyter/lab
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📊 Sample Code

### Text-to-Image Generation
```python
from diffusers import StableDiffusionPipeline
import torch

model_path = '/jupyterlab/hf/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b'
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, safety_checker=None)
pipe = pipe.to('cuda')

image = pipe("a fantasy landscape with mountains and rivers", width=512, height=512).images[0]
image.save("generated_landscape.png")
```

> `TODO`: Make the follwing example explicit with real images.

### Image Inpainting
```python
from diffusers import StableDiffusionInpaintPipeline
import torch

model_path = '/jupyterlab/hf/hub/models--fluently--Fluently-v3-inpainting/snapshots/9983a35adbd27ba2a56b392e46b237535af176a8'
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16, safety_checker=None)
pipe = pipe.to('cuda')

# Use with your base image and mask
result = pipe(prompt="a beautiful garden", image=base_image, mask_image=mask).images[0]
```

## 📝 Included Notebooks

1. **`stable-diffusion-v1-4-demo.ipynb`**: Complete tutorial for text-to-image generation
   - Generate multiple cat images
   - Create fantasy landscapes
   - Save and display results

2. **`fluently-v3-inpainting-demo.ipynb`**: Comprehensive inpainting examples
   - Create demo scenes
   - Apply different inpainting prompts
   - Advanced masking techniques

## 🔧 System Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **Memory**: Minimum 12GB GPU VRAM recommended
- **Docker**: Docker with GPU support (nvidia-docker2)

## 🏗️ Build Information

- **Base Image**: `sinanozel/kubyterlab-img:25.11`
- **Python Version**: 3.10
- **Key Dependencies**:
  - diffusers
  - transformers
  - accelerate
  - safetensors
  - xformers
  - torch (with CUDA support)

## 📄 License

This container is licensed under MIT. However, please note:

- **CompVis Stable Diffusion v1.4**: CreativeML Open RAIL-M License
- **Fluently v3 Inpainting**: CreativeML Open RAIL-M License

Please review the individual model licenses before commercial use.

## 🤝 Contributing

This image is part of the [jupyterlab-on-kubernetes](https://github.com/sinan-ozel/jupyterlab-on-kubernetes) project. Contributions welcome!

## ⚠️ Responsible AI Usage

These models are powerful tools for creative work. Please use them responsibly:
- Respect copyright and intellectual property
- Avoid generating harmful or inappropriate content
- Follow the CreativeML Open RAIL-M license terms
- Consider the ethical implications of AI-generated content
- Please make it clear to others that you used AI. People can tell if they look closely anyway, so even if you misrepresent your own contribution, it will be eventually obvious.

---

For more information, visit the [main project repository](https://github.com/sinan-ozel/jupyterlab-on-kubernetes).
