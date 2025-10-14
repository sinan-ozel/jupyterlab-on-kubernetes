# IaC For Generative AI: LLM Jupyterlab on Kubernetes on AWS

I am using this repo for my JupyterLab use cases. I achieve four goals:

1. I can deploy this server on a Kubernetes cluster, meaning that I achieve a level of cloud-independence. Instead of being locked-in to AWS using SageMaker, I create my own container, and I can deploy it in multiple cloud providers, since they all support Kubernetes.
2. I freeze my versions, meaning that I can replicate the environment in another container with relative ease when I want to package something I created.
3. I want to ensure compability with CUDA drivers and tensorflow
4. Combined with the IaC repo, I can deploy and completely clean-up my environment within minutes, meaning that I can keep my data sci

When things got to the point where almost all of the newer technology is on the cloud, I had to create this solution so that I can experiment easily, professionally and on a budget.

This is what the environment looks like in the end.
![1_SfmiSe5NHwsgVJgbDgg_Kw](https://github.com/user-attachments/assets/3566e9a5-30e6-4871-80b3-e527cd72a1c4)

## ⚙️ Features

### 🧱 Base Image
- **nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04** — CUDA 12.9.1 with cuDNN on Ubuntu 24.04 for GPU acceleration.

---

### 💻 Core Languages & Tools
- 🐍 **Python 3.12** + `pip`
- 🟩 **Node.js** + `npm`
- 🧰 **Build essentials:** `gcc`, `make`, `pkg-config`, `libtool`, `apt-utils`

---

### 👁️ OCR Engine
- 🔤 **Tesseract OCR** built from the latest `main` branch source
- 🧩 Includes **Leptonica** and related dependencies
- 🌐 Installs **language data** via bundled `get-languages.sh` script

---

### 📓 Jupyter Environment
- 🧪 **JupyterLab 4.4.9** with:
  - `ipywidgets 8.1.7`
  - `jupyter_contrib_nbextensions 0.7.0`
  - TOC extension pre-enabled
- 📡 Runs on **port 8888** (`--ip=0.0.0.0`, root allowed)

---

### 🧠 Deep Learning Frameworks
- 🔥 **PyTorch 2.8.0**
- 🧬 **TensorFlow 2.20.0** (CUDA-enabled)
- ⚙️ **tf-keras 2.20.1**

---

### 🗣️ NLP & LLM Toolkits
- 🤖 **Transformers 4.56.2**, **Diffusers 0.35.1**, **Accelerate 1.10.1**
- ⚡ **BitsAndBytes**, **FlashAttention**, **PEFT**
- 🧩 **LangChain (core + community + experimental + huggingface)**
- 🧠 **Sentence-Transformers**, **FastEmbed**, **FlashRank**, **ReRankers**, **GLiNER**, **Rank-BM25**, **IR-Measures**

---

### 📄 Document & Data Handling
- 📚 **PyPDF 6.1.1**, **OpenParse 0.7.0**
- 🧮 **LanceDB 0.25.1** for vector storage and retrieval

---

### 🧪 Evaluation & Utility Packages
- 🧭 **DeepEval 3.6.4** — LLM evaluation
- 📝 **MarkItDown 0.1.3** — Markdown conversion and processing
- 🔡 **mangoCR 0.1.4** — Character recognition
- 🧩 **ollm 0.5.0**, **ollmcp 0.18.2** — Model orchestration

---

### 🧰 Additional Utilities
- 🌐 `curl`, `wget`, `rsync`, `zip`, `unzip`, `ca-certificates`
- 🧹 Cleans up apt caches and temporary files to reduce image size

---

### 🚀 Runtime Command
```bash
jupyter lab --ip=0.0.0.0 --port=8888 \
  --no-browser \
  --ServerApp.root_dir='/jupyterlab/notebooks' \
  --allow-root



## Requirements

* Docker: You need to have this locally. If you have a Windows machine, you will also need WSL 2 running to be able to run the Linux containers.
* An AWS account and AWS CLI installed: You will need the account id and the account secret to login, push an image, and finally to deploy on Kubernetes on AWS. Alternatively, use Google, Azure, or Exoscale, basically any provider that has Kubernetes support. Alternatively, if you run it locally, you can use this with NVIDIA Graphics adapters.
* (Optional) VS Code - the steps are automated, making it easy to build and push the required image.

## Examples

TODO: docker-compose examples will follow.

## Version Support

### 25.09

TODO

Tested with:
1. WSL on Windows, Ubuntu 24.04, nvidia-smi 560.35.02

### 25.02

Tested with:
1. Locally: CUDA 12.6 on WSL on Windows, Ubuntu 22.04, nvidia-smi 560.35.02
2. AWS: AMI `AL2_x86_64_GPU` on EC2 type `g4dn.2xlarge`

[pip freeze](kubyterlab-llm/freeze/25.02.txt)

## Usage (Only needed if you want to customize)

Build, freeze the versions

## Version Compatibility

TODO