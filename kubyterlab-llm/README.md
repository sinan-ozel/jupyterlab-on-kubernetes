# 🤖 Kubyterlab-LLM - GPU-Accelerated LLM Development Environment

> 🚀 A production-ready JupyterLab environment for Large Language Model development with CUDA 12.8, PyTorch, TensorFlow, and comprehensive OCR support.

## 🎯 Purpose

**Kubyterlab-LLM** provides a complete GPU-accelerated environment for LLM research and production, achieving four key goals:

1. **☸️ Cloud Independence** - Deploy on any Kubernetes cluster across AWS, GCP, Azure, or on-premises. Not locked into AWS SageMaker or any single cloud provider.
2. **🔒 Version Freezing** - All dependencies are pinned for reproducible environments. Freeze files available for each version to replicate exact package states.
3. **⚡ CUDA Compatibility** - Ensures compatibility between CUDA drivers, TensorFlow, and PyTorch for seamless GPU acceleration.
4. **⚙️ Rapid Deployment** - Combined with IaC, deploy and completely tear down your environment within minutes, enabling budget-friendly experimentation.

This approach follows the "develop locally, deploy globally" philosophy, ensuring your notebooks work consistently from laptop to cloud.

## ✅ Tested Environments

| Environment | Platform | GPU | VRAM | Driver | CUDA | Status |
|-------------|----------|-----|------|--------|------|--------|
| **AWS EKS** | `g4dn` instance | Tesla T4 | 16GB | AL2023_x86_64_NVIDIA AMI | 12.8 | ✅ Production |
| **Windows WSL** | Ubuntu 24.04 | GeForce RTX 3060 | 12GB | 581.29 (NVML 580.82) | 13.0 | ✅ Development |
| **Ubuntu Desktop** | Ubuntu 24.04 | GeForce GTX 1660 Ti | 6GB | 580.95 | 13.0 | ✅ Development |

> **Note**: While the image targets CUDA 12.8, it's compatible with CUDA 13.0+ drivers thanks to forward compatibility.

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-sinanozel%2Fkubyterlab--llm-blue?logo=docker)](https://hub.docker.com/r/sinanozel/kubyterlab-llm)
[![Docker Pulls](https://img.shields.io/docker/pulls/sinanozel/kubyterlab-llm)](https://hub.docker.com/r/sinanozel/kubyterlab-llm)
[![Docker Image Size](https://img.shields.io/docker/image-size/sinanozel/kubyterlab-llm/25.11)](https://hub.docker.com/r/sinanozel/kubyterlab-llm)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Docker Pulls 25.11](https://img.shields.io/docker/pulls/sinanozel/kubyterlab-llm?label=docker%20pulls%2025.11)](https://hub.docker.com/r/sinanozel/kubyterlab-llm)

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [GPU Requirements](#-gpu-requirements)
- [Installed Frameworks](#-installed-frameworks)
- [Usage Examples](#-usage-examples)
- [Docker Compose Setup](#-docker-compose-setup)
- [Version History](#-version-history)

## ✨ Features

### 🧱 Base Image
- **nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04** — CUDA 12.8 with cuDNN on Ubuntu 24.04 for GPU acceleration

### 💻 Core Languages & Tools
- 🐍 **Python 3.12** + `pip`
- 🟩 **Node.js** + `npm`
- 🧰 **Build essentials**: `gcc`, `g++`, `make`, `pkg-config`, `libtool`
- 🎬 **FFmpeg** for multimedia processing

### 👁️ OCR Engine
- 🔤 **Tesseract OCR 5.5.1** built from source with latest optimizations
- 🧩 Includes **Leptonica** and related dependencies
- 🌐 Language data for **100+ languages** via bundled script
- 📄 Multiple OCR backends: EasyOCR, PaddleOCR, OLMOCR, MangoCR

### 📓 Jupyter Environment
- 🧪 **JupyterLab 4.4.9** with:
  - `ipywidgets 8.1.7` for interactive widgets
  - `jupyter_contrib_nbextensions 0.7.0` for enhanced functionality
  - Table of Contents extension pre-enabled
- 📡 Runs on **port 8888** (`--ip=0.0.0.0`, root allowed)
- 🔧 Configured for optimal notebook experience

### 🎮 GPU & Core Frameworks
- 🔥 **PyTorch 2.8.0** with CUDA support
- 🧬 **TensorFlow 2.20.0** with CUDA acceleration
- ⚙️ **TF-Keras 2.20.1** for high-level API
- ⚡ **Flash Attention 2.8.3** for efficient transformer attention
- 🧩 **xFormers 0.0.32** for memory-efficient transformers

### 🤗 Hugging Face Ecosystem
- 🤖 **Transformers 4.56.2** - State-of-the-art NLP models
- 🎨 **Diffusers 0.35.1** - Stable Diffusion and image generation
- 📊 **Datasets 3.0.0** - Efficient data loading and processing
- 🚀 **Accelerate 1.10.1** - Multi-GPU training utilities
- 🔧 **PEFT 0.17.1** - Parameter-efficient fine-tuning (LoRA, QLoRA)
- 💾 **BitsAndBytes 0.48.1** - 8-bit and 4-bit quantization

### 🔍 RAG & Retrieval
- 🦜 **LangChain 0.3.27** - LLM application framework
  - `langchain-community 0.3.30`
  - `langchain-experimental 0.3.4`
  - `langchain-huggingface 0.3.1`
- 🧠 **DSPy 3.0.3** - Programming with foundation models
- 📐 **Sentence Transformers 5.1.1** - Text embeddings
- ⚡ **FastEmbed 0.7.3** - Lightweight embeddings
- 🔥 **FlashRank 0.2.10** - Fast reranking
- 🎯 **Rerankers 0.10.0** - Advanced reranking models
- 📊 **Rank-BM25 0.2.2** - Classic information retrieval
- 🔬 **IR-Measures 0.4.1** - Information retrieval metrics
- 🏷️ **GLiNER 0.2.22** - Generalist named entity recognition

### 📄 OCR & Document Processing
- 🔤 **Tesseract 5.5.1** - Industry-standard OCR (100+ languages)
- 👁️ **EasyOCR 1.7.2** - Deep learning OCR
- 🐼 **PaddleOCR 3.3.1** - Multi-lingual OCR with PaddlePaddle GPU backend
- 📱 **OLMOCR 0.4.4** - Advanced document OCR
- 🥭 **MangoCR 0.1.4** - Document understanding
- 📐 **LayoutParser 0.3.4** - Document layout analysis
- 🖼️ **PDF2Image 1.16.3** - PDF to image conversion
- 📝 **PyPDF 6.1.1** - PDF text extraction
- 📖 **OpenParse 0.7.0** - Advanced PDF parsing
- 📄 **Markitdown 0.1.3** - Markdown conversion with support for all formats
- 🖼️ **Pillow-HEIF 1.1.1** - HEIC/HEIF image support

### 🗄️ Vector Databases
- 🗃️ **LanceDB 0.25.1** - Apache Arrow-based vector store
- 🔧 **Pylance 0.38.1** - Lance format utilities

### 🛠️ Additional Tools
- 🤝 **Atomic Agents 2.2.0** - Agent framework
- 🔄 **LiteLLM 1.79.1** - Unified LLM API
- 📊 **DeepEval 3.6.4** - LLM evaluation framework
- 🦙 **OLLM 0.5.0** & **OLLMCP 0.18.2** - Ollama integration
- 📚 **NLTK 3.9.2** - Natural language toolkit
- 🤗 **Hugging Face Hub 0.35.3** - Model hub integration

### 🧹 Utilities
- 🌐 `curl`, `wget`, `rsync`, `zip`, `unzip`, `ca-certificates`
- 🧼 Optimized layer caching and cleanup for reduced image size

## 🚀 Quick Start

### Prerequisites

- **Docker Engine 20.10+**
- **NVIDIA GPU** with CUDA 12.x support
- **NVIDIA Docker Runtime** (nvidia-docker2)
- **16GB+ GPU memory** recommended
- **32GB+ system RAM** recommended

### Verify GPU Support

```bash
# Check NVIDIA driver
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### Pull and Run

```bash
# Pull the image
docker pull sinanozel/kubyterlab-llm:25.11

# Run with GPU support
docker run -d \
  --gpus all \
  --name jupyterlab-llm \
  -p 8888:8888 \
  -v $(pwd)/notebooks:/jupyterlab/notebooks \
  sinanozel/kubyterlab-llm:25.11

# Get access token
docker exec jupyterlab-llm jupyter server list

# Access at http://localhost:8888
```

### Runtime Command

The container starts JupyterLab with the following command:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 \
  --no-browser \
  --ServerApp.root_dir='/jupyterlab/notebooks' \
  --allow-root
```

## 🎮 GPU Requirements

### Minimum Specifications
- **CUDA Compute Capability**: 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- **VRAM**: 8GB minimum, 16GB+ recommended
- **Driver Version**: 525+ (for CUDA 12.x)

### Tested GPUs
- ✅ NVIDIA RTX 3090/4090
- ✅ NVIDIA A100/H100
- ✅ NVIDIA T4
- ✅ NVIDIA V100

## 📦 Installed Frameworks

### Deep Learning Frameworks

| Framework | Version | Purpose |
|-----------|---------|---------|
| PyTorch | 2.8.0 | Primary deep learning framework |
| TensorFlow | 2.20.0 | Alternative DL framework |
| TF-Keras | 2.20.1 | High-level Keras API |
| Flash Attention | 2.8.3 | Efficient attention mechanism |
| xFormers | 0.0.32 | Memory-efficient transformers |

### Transformer Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Transformers | 4.56.2 | Hugging Face models |
| Diffusers | 0.35.1 | Stable Diffusion pipeline |
| Sentence Transformers | 5.1.1 | Text embeddings |
| PEFT | 0.17.1 | LoRA/QLoRA fine-tuning |
| BitsAndBytes | 0.48.1 | Model quantization |

### OCR Engines

| Engine | Version | Languages |
|--------|---------|-----------|
| Tesseract | 5.5.1 | 100+ |
| EasyOCR | 1.7.2 | 80+ |
| PaddleOCR | 3.3.1 | Multi-lingual |
| OLMOCR | 0.4.4 | Document-focused |

## 💡 Usage Examples

### 🔥 PyTorch GPU Check

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test computation
x = torch.randn(1000, 1000, device='cuda')
y = torch.matmul(x, x)
print("✅ GPU computation successful!")
```

### 🤗 Load a Transformer Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model in 8-bit
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16
)

# Generate text
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 🎨 Stable Diffusion Image Generation

```python
from diffusers import StableDiffusionPipeline
import torch

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",

# Generate image
prompt = "A serene landscape with mountains and a lake, digital art"
image = pipe(prompt).images[0]
image.save("generated_landscape.png")
```

### 🔧 LoRA Fine-Tuning with PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Model is now ready for fine-tuning with reduced memory
```

### 📝 Tesseract OCR

```python
import pytesseract
from PIL import Image

# Load image
image = Image.open("document.png")

# Extract text
text = pytesseract.image_to_string(image, lang='eng')
print(text)

# Get bounding boxes
boxes = pytesseract.image_to_boxes(image)
print(f"Detected {len(boxes.split('\\n'))} characters")

# Get detailed data
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
```

### 🔍 RAG Pipeline with LangChain

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Load documents
documents = ["Document 1 text...", "Document 2 text..."]

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.create_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vectorstore = FAISS.from_documents(splits, embeddings)

# Create LLM
llm_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    device=0
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Ask questions
result = qa_chain.run("What is the main topic of these documents?")
print(result)
```

### 🗄️ LanceDB Vector Search

```python
import lancedb
import numpy as np

# Connect to database
db = lancedb.connect("./lancedb_data")

# Create table with vectors
data = [
    {
        "vector": np.random.randn(384).tolist(),
        "text": "First document",
        "metadata": {"source": "web"}
    },
    {
        "vector": np.random.randn(384).tolist(),
        "text": "Second document",
        "metadata": {"source": "api"}
    }
]

table = db.create_table("documents", data=data)

# Search
query_vector = np.random.randn(384).tolist()
results = table.search(query_vector).limit(5).to_pandas()
print(results)
```

### 📊 Model Evaluation with DeepEval

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Define test case
test_case = LLMTestCase(
    input="What is machine learning?",
    actual_output="Machine learning is a subset of AI that enables systems to learn from data.",
    expected_output="Machine learning allows computers to learn without explicit programming."
)

# Evaluate
metric = AnswerRelevancyMetric(threshold=0.7)
metric.measure(test_case)
print(f"Score: {metric.score}")
print(f"Passed: {metric.is_successful()}")
```

## 🐳 Docker Compose Setup

Create `docker-compose.yaml`:

```yaml
version: '3.8'

services:
  jupyterlab-llm:
    image: sinanozel/kubyterlab-llm:25.11
    container_name: jupyterlab-llm
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/jupyterlab/notebooks
      - ./models:/root/.cache/huggingface
      - ~/.ssh:/root/.ssh:ro
    environment:
      - JUPYTER_ENABLE_LAB=yes
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    shm_size: '16gb'
    restart: unless-stopped

networks:
  default:
    driver: bridge
```

Start the environment:

```bash
docker-compose up -d
docker-compose logs -f
```

## 📚 Version History

### Version 25.11 (Current)

**Status**: ✅ Production Ready

**Tested Environments**:
- AWS EKS on `g4dn` instances with Tesla T4 GPUs
- Windows WSL Ubuntu 24.04 with GeForce RTX 3060 (12GB)
- Ubuntu 24.04 Desktop with GeForce GTX 1660 Ti (6GB)

**Key Updates**:
- CUDA 12.8 with cuDNN
- PyTorch 2.8.0
- TensorFlow 2.20.0
- Comprehensive OCR support (Tesseract 5.5.1, PaddleOCR, EasyOCR)
- Latest Hugging Face ecosystem packages
- [Pip Freeze Output](freeze/25.11.txt)

### Version 25.10

**Tested Environments**:
- CUDA 13.0 on WSL Windows, Ubuntu 24.04 (GeForce RTX 3060, 12GB)
- CUDA 13.0 on Ubuntu 24.04 Desktop (GeForce GTX 1660 Ti, 6GB)

**Updates**:
- Initial CUDA 13.0 compatibility testing
- Updated transformer libraries
- [Pip Freeze Output](freeze/25.10.txt)

### Version 25.09

**Tested Environments**:
- CUDA 13.0 on WSL Windows, Ubuntu 24.04 (GeForce RTX 3060, 12GB)

**Updates**:
- Enhanced LangChain integration
- Added DSPy framework
- [Pip Freeze Output](freeze/25.09.txt)

### Version 25.02

**Tested Environments**:
- Local: CUDA 12.6 on WSL Windows, Ubuntu 22.04 (GeForce RTX 3060, 12GB)
- AWS: AMI `AL2_x86_64_GPU` on EC2 type `g4dn.2xlarge`

**Updates**:
- Initial release with comprehensive ML stack
- Tesseract OCR integration
- [Pip Freeze Output](freeze/25.02.txt)

## ⚙️ Configuration

### Hugging Face Authentication

```python
from huggingface_hub import login

# Login to access gated models
login(token="your_hf_token_here")
```

Or set environment variable:

```bash
export HF_TOKEN="your_hf_token_here"
```

### Memory Management

```python
import torch

# Clear GPU cache
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## 🧹 Cleanup

```bash
# Stop container
docker stop jupyterlab-llm

# Remove container
docker rm jupyterlab-llm

# Clear GPU memory
nvidia-smi --gpu-reset
```

## 📚 Available Languages (Tesseract)

The image includes trained data for 100+ languages:
- **Western**: English, French, German, Spanish, Italian, Portuguese
- **Eastern European**: Russian, Polish, Czech, Ukrainian
- **Asian**: Chinese (Simplified/Traditional), Japanese, Korean, Arabic, Hindi
- **And many more...**

Full list in `kubyterlab-llm/tesseract/languages.txt`

## 🤝 Contributing

Contributions are welcome! Please submit Pull Requests for:
- Additional ML frameworks
- OCR engine improvements
- Documentation enhancements
- Performance optimizations

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Built on top of:
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch](https://pytorch.org/)
- [Hugging Face](https://huggingface.co/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Jupyter Docker Stacks](https://github.com/jupyter/docker-stacks)

---

Made with ❤️ for ML engineers and researchers working with LLMs
