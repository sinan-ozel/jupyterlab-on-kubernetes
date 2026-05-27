# 🚀 JupyterLab on Kubernetes

> Production-ready JupyterLab environments for data science and LLM development. Deploy locally with Docker Compose or scale to Kubernetes clusters across any cloud provider.

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker Pulls JupyterLab LLM](https://img.shields.io/docker/pulls/sinanozel/kubyterlab-llm?label=docker%20pulls%20kubyterlab-llm)](https://hub.docker.com/r/sinanozel/kubyterlab-llm)
[![Docker Pulls JupyterLab Image](https://img.shields.io/docker/pulls/sinanozel/kubyterlab-img?label=docker%20pulls%20kubyterlab-img)](https://hub.docker.com/r/sinanozel/kubyterlab-img)
[![Docker Pulls JupyterLab Image with <12GB VRAM Models](https://img.shields.io/docker/pulls/sinanozel/kubyterlab-img-12g?label=docker%20pulls%20kubyterlab-img-12g)](https://hub.docker.com/r/sinanozel/kubyterlab-img-12g)
[![Docker](https://img.shields.io/badge/Docker-Required-2496ED?logo=docker)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5?logo=kubernetes)](https://kubernetes.io/)

## 🎯 Overview

This repository provides four production-ready JupyterLab environments designed for the modern ML workflow:

1. **[Kubyterlab-DS](kubyterlab-ds/README.md)** - Data Science & RAG Development
2. **[Kubyterlab-LLM](kubyterlab-llm/README.md)** - GPU-Accelerated LLM Development
3. **[Kubyterlab-IMG](kubyterlab-img/kubyterlab-img/README.md)** - Base Image Generation Environment
4. **[Kubyterlab-IMG-12G](kubyterlab-img/kubyterlab-img-12g/README.md)** - Pre-loaded Stable Diffusion Models

Both environments follow the **"develop locally, deploy globally"** philosophy:
- 💻 **Local Development**: Run complete stacks with `docker-compose` for rapid prototyping
- ☸️ **Kubernetes Deployment**: Deploy to any K8s cluster (AWS EKS, GCP GKE, Azure AKS, on-premises)
- 🔒 **Production Ready**: Pinned dependency versions with freeze files for reproducibility
- 🌐 **Cloud Independent**: Not locked into AWS SageMaker or any single cloud provider

![JupyterLab Environment](https://github.com/user-attachments/assets/3566e9a5-30e6-4871-80b3-e527cd72a1c4)

## 📦 Environments

### 🧪 [Kubyterlab-DS](kubyterlab-ds/README.md) - Data Science Environment

**Perfect for**: Data analysis, RAG pipelines, web scraping, document processing, and vector database experimentation.

**Key Features**:
- 📊 Core data science stack: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- 🔴 Redis for caching and queues
- 🎯 Qdrant vector database
- 🧠 ChromaDB & LanceDB embedded vector stores
- 🤖 Ollama embedding & LLM services
- 🌐 Web scraping tools: BeautifulSoup4, Requests, LXML
- 📄 Document processing: PyPDF, OpenPyXL, PyYAML

**Quick Start**:
```bash
docker pull sinanozel/kubyterlab-ds:25.11
```

[📖 Full Documentation](kubyterlab-ds/README.md) | [🐳 Docker Hub](https://hub.docker.com/r/sinanozel/kubyterlab-ds)

---

### 🤖 [Kubyterlab-LLM](kubyterlab-llm/README.md) - GPU-Accelerated LLM Environment

**Perfect for**: Transformer model fine-tuning, LLM training, image generation, OCR at scale, and multi-modal AI applications.

**Key Features**:
- 🎮 **CUDA 12.8** with cuDNN for GPU acceleration
- 🔥 **PyTorch 2.8.0** & **TensorFlow 2.20.0**
- 🤗 Complete Hugging Face ecosystem: Transformers, Diffusers, PEFT, BitsAndBytes
- ⚡ Flash Attention 2.8.3 & xFormers for efficient transformers
- 🦜 LangChain & DSPy for LLM application development
- 👁️ Comprehensive OCR: Tesseract 5.5.1, PaddleOCR, EasyOCR, OLMOCR (100+ languages)
- 📊 Evaluation frameworks: DeepEval, IR-Measures
- 🗄️ LanceDB for vector storage

**Quick Start**:
```bash
docker pull sinanozel/kubyterlab-llm:25.11
docker run -d --gpus all -p 8888:8888 sinanozel/kubyterlab-llm:25.11
```

[📖 Full Documentation](kubyterlab-llm/README.md) | [🐳 Docker Hub](https://hub.docker.com/r/sinanozel/kubyterlab-llm)

---

### 🎨 [Kubyterlab-IMG](kubyterlab-img/kubyterlab-img/README.md) - Base Image Generation Environment

**Perfect for**: Custom AI image generation setups, foundation for building specialized containers with your own models.

**Key Features**:
- 🎮 **CUDA-enabled** base environment for GPU acceleration
- 🤗 Diffusers pipeline for Stable Diffusion
- 🔧 Flexible foundation - bring your own models
- 📝 Sample notebooks for common workflows

**Quick Start**:
```bash
docker pull sinanozel/kubyterlab-img:25.11
```

[📖 Full Documentation](kubyterlab-img/kubyterlab-img/README.md) | [🐳 Docker Hub](https://hub.docker.com/r/sinanozel/kubyterlab-img)

---

### 🖼️ [Kubyterlab-IMG-12G](kubyterlab-img/kubyterlab-img-12g/README.md) - Pre-loaded Stable Diffusion Models

**Perfect for**: Immediate AI image generation, text-to-image workflows, image inpainting tasks without model downloads.

**Key Features**:
- 🚀 **Pre-installed Models**: CompVis Stable Diffusion v1.4 & Fluently v3 Inpainting
- ⚡ **Ready-to-Use**: No model download required - start generating immediately
- 🎯 **12GB Models**: High-quality image generation capabilities
- 📚 **Sample Notebooks**: Complete examples for both text-to-image and inpainting
- 🏷️ **Proper Attribution**: Model citations embedded in container labels

**Quick Start**:
```bash
docker pull sinanozel/kubyterlab-img-12g:25.11
docker run --gpus all -p 8888:8888 sinanozel/kubyterlab-img-12g:25.11
```

[📖 Full Documentation](kubyterlab-img/kubyterlab-img-12g/README.md) | [🐳 Docker Hub](https://hub.docker.com/r/sinanozel/kubyterlab-img-12g)

---

## 🎯 Use Cases

### Kubyterlab-DS Use Cases

✅ **Data Analysis & Visualization**
- Exploratory data analysis with pandas and matplotlib
- Statistical analysis and hypothesis testing
- Interactive dashboards and reports

✅ **RAG Application Development**
- Build retrieval-augmented generation pipelines
- Test different vector databases (Qdrant, ChromaDB, LanceDB)
- Experiment with embedding models and rerankers

✅ **Web Scraping & Data Collection**
- Scrape data from websites with BeautifulSoup4
- Process Excel, PDF, and YAML files
- Store and cache data in Redis

✅ **Prototype → Production Workflow**
- Develop locally with docker-compose
- Test with embedded LLM services (Ollama)
- Deploy to Kubernetes with minimal changes

### Kubyterlab-LLM Use Cases

✅ **LLM Fine-Tuning & Training**
- Fine-tune models with LoRA/QLoRA using PEFT
- 8-bit and 4-bit quantization with BitsAndBytes
- Multi-GPU training with Accelerate

✅ **Image Generation & Multi-Modal AI**
- Stable Diffusion image generation
- Text-to-image and image-to-image pipelines
- Vision-language model experiments

✅ **Document Understanding at Scale**
- OCR in 100+ languages with Tesseract
- Layout analysis with LayoutParser
- PDF processing and information extraction

✅ **Production LLM Applications**
- Develop with LangChain and DSPy
- Evaluate with DeepEval
- Deploy to GPU-enabled Kubernetes clusters

---

## 🚀 Deployment Options

### Local Development (Docker Compose)

Both environments include `docker-compose.yaml` configurations for local development with all supporting services.

**Kubyterlab-DS** includes:
- JupyterLab
- Redis
- Qdrant vector database
- Ollama embedding service
- Ollama LLM service

**Kubyterlab-LLM** runs standalone:
- JupyterLab with GPU access
- Volume mounts for notebooks and Hugging Face models

### Kubernetes Deployment

Deploy to any Kubernetes cluster with GPU support:

**Tested Environments**:
- ✅ AWS EKS with `g4dn` instances
- ✅ Custom Kubernetes clusters
- ✅ On-premises with NVIDIA GPUs

See individual READMEs for Kubernetes manifests and Helm charts.

---

## 📋 Requirements

### For Kubyterlab-DS
- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum

### For Kubyterlab-LLM
- Docker Engine 20.10+
- NVIDIA GPU with CUDA 12.x support
- NVIDIA Docker Runtime (nvidia-docker2)
- 16GB+ GPU memory recommended
- 32GB+ system RAM recommended

### For Kubernetes Deployment
- Kubernetes 1.24+
- kubectl configured
- For LLM: GPU nodes with NVIDIA device plugin

---

## 🔄 Version Management

Each version includes frozen dependency lists for reproducibility:

**Kubyterlab-DS**:
- [25.12 freeze file](kubyterlab-ds/freeze/25.12.txt)
- [25.11 freeze file](kubyterlab-ds/freeze/25.11.txt)

**Kubyterlab-LLM**:
- [26.01 freeze file](kubyterlab-llm/freeze/26.01.txt)
- [25.11 freeze file](kubyterlab-llm/freeze/25.11.txt)
- [25.10 freeze file](kubyterlab-llm/freeze/25.10.txt)
- [25.09 freeze file](kubyterlab-llm/freeze/25.09.txt)
- [25.02 freeze file](kubyterlab-llm/freeze/25.02.txt)

---

## 🛠️ Development Tools

This repository includes VS Code development container configuration and tasks for:

- Building Docker images with OCI-compliant labels
- Running automated tests
- Freezing dependency versions
- Pushing to Docker Hub
- Pushing to AWS ECR
- Deploying to Kubernetes

See [.vscode/tasks.json](.vscode/tasks.json) for available automation tasks.

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Additional ML frameworks and libraries
- Performance optimizations
- Documentation improvements
- Kubernetes deployment examples
- CI/CD pipeline enhancements

Please submit Pull Requests with detailed descriptions.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

Built on top of:
- [Jupyter Docker Stacks](https://github.com/jupyter/docker-stacks)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch](https://pytorch.org/)
- [Hugging Face](https://huggingface.co/)
- [Redis](https://redis.io/)
- [Qdrant](https://qdrant.tech/)
- [Ollama](https://ollama.ai/)

---

Made with ❤️ for data scientists and ML engineers working with modern AI infrastructure
