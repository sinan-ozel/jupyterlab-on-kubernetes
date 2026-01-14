FROM nvidia/cuda:12.5.1-base-ubuntu22.04

# Set labels for the image
LABEL org.opencontainers.image.title="kubyterlab-img"
LABEL org.opencontainers.image.description="JupyterLab base image for AI image generation with CUDA support"
LABEL org.opencontainers.image.version="25.12"
LABEL org.opencontainers.image.authors="Sinan Ozel"
LABEL org.opencontainers.image.url="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img"
LABEL org.opencontainers.image.source="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img"
LABEL org.opencontainers.image.vendor="Sinan Ozel"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.base.name="nvidia/cuda:12.5.1-base-ubuntu22.04"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    nodejs \
    npm \
    python3.11 \
    python3-pip \
    sudo \
    python3.11-venv \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Create jovyan user with passwordless sudo
RUN useradd -m -s /bin/bash -N -u 1000 jovyan && \
    echo "jovyan ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/jovyan && \
    chmod 0440 /etc/sudoers.d/jovyan

# Ensure user-installed scripts are on PATH for subsequent RUNs
ENV PATH=/home/jovyan/.local/bin:$PATH

# Switch to jovyan user for remaining operations
USER jovyan

# Install JupyterLab and core notebook packages
RUN pip install --no-cache-dir \
    jupyterlab==4.4.9 \
    ipywidgets==8.1.7 \
    ipykernel==7.1.0 \
    jupyterlab-git==0.51.3 \
    jupyter-toc==1.1.2 \
    sidecar==0.8.0 \
    jupyterlab-code-formatter==3.0.2 \
    jupyter_contrib_nbextensions==0.7.0

# Install deep learning frameworks
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    tensorflow[and-cuda]==2.20.0 \
    tensorrt

# RUN pip3 install git+https://github.com/huggingface/transformers
RUN pip3 install transformers==4.57.1
# RUN pip3 install diffusers[torch]==0.35.2
RUN pip install git+https://github.com/huggingface/diffusers.git@b4e6dc3037e75e2dc16466b914b1be597b06be9a

# Install AI/ML supporting libraries and data science packages
RUN pip install --no-cache-dir \
    accelerate==1.10.1 \
    xformers==0.0.32.post2 \
    tqdm==4.67.1 \
    # torchvision==0.24.1 \
    bitsandbytes==0.48.2 \
    numpy==2.2.6 \
    pandas==2.3.3 \
    pillow==12.0.0 \
    pillow-heif==1.1.1 \
    matplotlib==3.10.8 \
    seaborn==0.13.2

EXPOSE 8888
WORKDIR /jupyterlab
CMD ["python3", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir='/jupyterlab/notebooks'", "--allow-root"]
