FROM nvidia/cuda:12.9.2-base-ubuntu24.04

# Set labels for the image
LABEL org.opencontainers.image.title="kubyterlab-img"
LABEL org.opencontainers.image.description="JupyterLab base image for AI image generation with CUDA support"
LABEL org.opencontainers.image.version="26.07"
LABEL org.opencontainers.image.authors="Sinan Ozel"
LABEL org.opencontainers.image.url="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img"
LABEL org.opencontainers.image.source="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img"
LABEL org.opencontainers.image.vendor="Sinan Ozel"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.base.name="nvidia/cuda:12.9.2-base-ubuntu24.04"

RUN apt-get update --fix-missing \
    && apt-get install --fix-missing -y \
    curl \
    git \
    ffmpeg \
    python3.12 \
    python3-pip \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 config set global.break-system-packages true

RUN pip3 install setuptools==82.0.1

RUN pip3 install jupyterlab==4.5.7
RUN pip3 install ipywidgets==8.1.8
RUN pip3 install jupyter_contrib_nbextensions==0.7.0
RUN jupyter labextension enable @jupyterlab/toc

# Always check xformer and torchvision compatiblity
RUN pip3 install torch==2.10.0
RUN pip3 install tensorflow[and-cuda]==2.20.0
RUN pip3 install tensorrt

RUN pip3 install torchvision==0.25.0

RUN pip3 install \
    diffusers==0.38.0 \
    accelerate==1.10.1 \
    safetensors==0.8.0rc0 \
    transformers==4.57.1 \
    opencv-python-headless==4.13.0.92 \
    xformers==0.0.35

RUN pip3 install tqdm==4.67.1
RUN pip3 install bitsandbytes==0.50.2
RUN pip3 install numpy==2.5.2
RUN pip3 install pandas==3.0.5
RUN pip3 install matplotlib==3.11.1
RUN pip3 install seaborn==0.13.2
RUN pip3 install insightface==1.0.1
RUN pip3 install gguf==0.19.0


RUN pip3 install \
    einops==0.8.2 \
    onnxruntime-gpu==1.26.0 \
    controlnet-aux==0.0.10

RUN git clone https://github.com/instantX-research/InstantID.git /jupyterlab/InstantID

EXPOSE 8888
WORKDIR /jupyterlab
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir='/jupyterlab/notebooks'", "--allow-root"]
CMD PYTHONPATH=/jupyterlab/InstantID:$PYTHONPATH jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.root_dir='/jupyterlab/notebooks' --allow-root

