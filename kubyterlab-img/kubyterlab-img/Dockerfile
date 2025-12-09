FROM nvidia/cuda:12.5.1-base-ubuntu22.04

# Set labels for the image
LABEL org.opencontainers.image.title="kubyterlab-img"
LABEL org.opencontainers.image.description="JupyterLab base image for AI image generation with CUDA support"
LABEL org.opencontainers.image.version="25.11"
LABEL org.opencontainers.image.authors="Sinan Ozel"
LABEL org.opencontainers.image.url="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img"
LABEL org.opencontainers.image.source="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img"
LABEL org.opencontainers.image.vendor="Sinan Ozel"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.base.name="nvidia/cuda:12.5.1-base-ubuntu22.04"

RUN apt update
RUN apt install -y curl
RUN apt install -y nodejs
RUN apt install -y npm
RUN apt install -y python3.11
RUN apt install -y python3-pip

RUN pip install jupyterlab==4.4.9
RUN pip install ipywidgets==8.1.7
RUN pip install jupyter_contrib_nbextensions==0.7.0
RUN jupyter labextension enable @jupyterlab/toc

RUN pip3 install torch==2.8.0
RUN pip3 install tensorflow[and-cuda]==2.20.0
RUN pip3 install tensorrt
# RUN pip3 install git+https://github.com/huggingface/transformers
RUN pip3 install transformers==4.57.1
# RUN pip3 install diffusers[torch]==0.35.2
RUN pip install git+https://github.com/huggingface/diffusers.git@b4e6dc3037e75e2dc16466b914b1be597b06be9a
RUN pip3 install accelerate==1.10.1
RUN pip3 install xformers==0.0.32.post2
RUN pip3 install tqdm==4.67.1
RUN pip3 install torchvision==0.24.1
RUN pip3 install bitsandbytes==0.48.2
RUN pip3 install numpy==2.3.5
RUN pip3 install pandas==2.3.3
RUN pip3 install matplotlib==3.10.7
RUN pip3 install seaborn==0.13.2

EXPOSE 8888
WORKDIR /jupyterlab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir='/jupyterlab/notebooks'", "--allow-root"]
