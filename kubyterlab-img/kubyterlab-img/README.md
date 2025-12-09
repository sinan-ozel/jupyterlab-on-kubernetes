# KubyterLab-IMG

A CUDA-enabled JupyterLab Docker image optimized for AI image generation and computer vision tasks. This base image provides the foundation for building custom AI image generation environments.

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-sinanozel%2Fkubyterlab--img-blue?logo=docker)](https://hub.docker.com/r/sinanozel/kubyterlab-img)
[![Docker Pulls](https://img.shields.io/docker/pulls/sinanozel/kubyterlab-img)](https://hub.docker.com/r/sinanozel/kubyterlab-img)
[![Docker Image Size](https://img.shields.io/docker/image-size/sinanozel/kubyterlab-img/25.11)](https://hub.docker.com/r/sinanozel/kubyterlab-img)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Features

- **GPU Acceleration**: NVIDIA CUDA 12.5.1 with full GPU support
- **Modern ML Stack**: PyTorch 2.8.0, TensorFlow 2.20.0, and TensorRT
- **Diffusers Ready**: Latest HuggingFace Diffusers for Stable Diffusion pipelines
- **Optimized Performance**: xFormers and BitsAndBytes for efficient model inference
- **Flexible Foundation**: Perfect base for building custom AI image generation containers

## 📦 Included Libraries

### Core ML Frameworks
- **PyTorch 2.8.0** - Deep learning framework with CUDA support
- **TensorFlow 2.20.0** - Google's ML platform with CUDA acceleration
- **TensorRT** - NVIDIA's inference optimization library

### AI Image Generation
- **Diffusers** (latest from GitHub) - HuggingFace's diffusion model library
- **Transformers 4.57.1** - State-of-the-art NLP and vision models
- **Accelerate 1.10.1** - Distributed training and inference
- **xFormers 0.0.32.post2** - Memory-efficient transformer implementations
- **BitsAndBytes 0.48.1** - 8-bit and 4-bit quantization

### Computer Vision
- **TorchVision 0.23.0** - Computer vision datasets and models
- **TQDM 4.67.1** - Progress bars for training loops

### JupyterLab Environment
- **JupyterLab 4.4.9** - Modern notebook interface
- **IPyWidgets 8.1.7** - Interactive widgets for notebooks
- **Jupyter Contrib NBExtensions 0.7.0** - Additional notebook extensions

## 🛠️ Usage

### Quick Start
```bash
# Pull the image
docker pull sinanozel/kubyterlab-img:25.11

# Run with GPU support
docker run --gpus all -p 8888:8888 -v $(pwd):/jupyterlab/notebooks sinanozel/kubyterlab-img:25.11
```

### With Docker Compose
```yaml
version: '3.8'
services:
  jupyter:
    image: sinanozel/kubyterlab-img:25.11
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

## 🎯 Use Cases

### Perfect For:
- **Custom AI Image Generation**: Build specialized containers with your own models
- **Stable Diffusion Development**: Develop and test diffusion model pipelines
- **Computer Vision Research**: CV model training and inference
- **Multi-Modal AI**: Projects combining text, image, and other modalities
- **Foundation for Specialized Images**: Base for creating domain-specific AI containers

### Example Workflows:
```python
from diffusers import StableDiffusionPipeline
import torch

# Load your own Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "your-model-path",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to('cuda')

# Generate images
image = pipe("a beautiful landscape").images[0]
image.save("generated_image.png")
```

## 🔧 System Requirements

- **GPU**: NVIDIA GPU with CUDA 12.x support
- **Memory**: Minimum 8GB GPU VRAM (16GB+ recommended for larger models)
- **Docker**: Docker with GPU support (nvidia-docker2)
- **System RAM**: 16GB+ recommended

## 🏗️ Build Information

- **Base Image**: `nvidia/cuda:12.5.1-base-ubuntu22.04`
- **Python Version**: 3.11
- **CUDA Version**: 12.5.1
- **Ubuntu Version**: 22.04 LTS

## 📊 Variants

This is the base image for the KubyterLab-IMG family:

- **[kubyterlab-img](./README.md)** (this image) - Base foundation with core libraries
- **[kubyterlab-img-12g](../kubyterlab-img-12g/README.md)** - Pre-loaded with Stable Diffusion models (12GB)
- **kubyterlab-img-32g** - Larger model variant (coming soon)

## 🚀 Extending This Image

Use this as a base for your custom AI image generation containers:

```dockerfile
FROM sinanozel/kubyterlab-img:25.11

# Copy your models
COPY ./models /jupyterlab/models

# Install additional dependencies
RUN pip install your-custom-packages

# Copy custom notebooks
COPY ./notebooks /jupyterlab/notebooks
```

## 🔄 Version Management

Current version includes frozen dependencies for reproducibility:
- [25.11 freeze file](freeze/25.11.txt)

## 📄 License

This container is licensed under MIT License.

## 🤝 Contributing

This image is part of the [jupyterlab-on-kubernetes](https://github.com/sinan-ozel/jupyterlab-on-kubernetes) project. Contributions welcome!

## 🙏 Acknowledgments

Built on top of:
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://tensorflow.org/)
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [Jupyter](https://jupyter.org/)

---

For more information, visit the [main project repository](https://github.com/sinan-ozel/jupyterlab-on-kubernetes).