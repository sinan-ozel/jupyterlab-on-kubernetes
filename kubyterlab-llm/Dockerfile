FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Set labels for the image
LABEL org.opencontainers.image.title="kubyterlab-llm"
LABEL org.opencontainers.image.description="JupyterLab environment for LLM development with CUDA support"
LABEL org.opencontainers.image.version="25.11"
LABEL org.opencontainers.image.authors="Sinan Ozel"
LABEL org.opencontainers.image.url="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-llm"
LABEL org.opencontainers.image.source="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-llm"
LABEL org.opencontainers.image.vendor="Sinan Ozel"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.base.name="nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04"

ARG TESSERACT_VERSION="5.5.1"
ARG TESSERACT_URL="https://api.github.com/repos/tesseract-ocr/tesseract/tarball/${TESSERACT_VERSION}"

RUN apt-get update && apt-get install -y \
    curl \
    nodejs \
    npm \
    python3.12 \
    python3-pip \
    wget \
    ca-certificates \
    apt-transport-https \
    g++ \
    make \
    libleptonica-dev \
    libtool \
    pkg-config \
    ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN wget -qO tesseract.tar.gz $TESSERACT_URL
RUN tar -xzf tesseract.tar.gz
RUN mv tesseract-* tesseract

WORKDIR /src/tesseract

RUN ./autogen.sh
RUN ./configure
RUN make
RUN make install
RUN ldconfig

RUN rm -rf /src

WORKDIR /usr/local/share/tessdata/

COPY kubyterlab-llm/tesseract/get-languages.sh .
COPY kubyterlab-llm/tesseract/languages.txt .
RUN chmod +x ./get-languages.sh
RUN ./get-languages.sh
RUN rm ./get-languages.sh
RUN rm ./languages.txt

RUN apt-get update && apt-get install -y \
    unzip \
    zip \
    rsync && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*


RUN pip3 install jupyterlab==4.4.9 --break-system-packages
RUN pip3 install ipywidgets==8.1.7 --break-system-packages
RUN pip3 install jupyter_contrib_nbextensions==0.7.0 --break-system-packages
RUN jupyter labextension enable @jupyterlab/toc

RUN pip3 install torch==2.8.0 --break-system-packages
RUN pip3 install tensorflow[and-cuda]==2.20.0 --break-system-packages
RUN pip3 install datasets==3.0.0 --break-system-packages

RUN pip3 install transformers==4.56.2 --break-system-packages
RUN pip3 install diffusers[torch]==0.35.1 --break-system-packages
RUN pip3 install accelerate==1.10.1 --break-system-packages
RUN pip3 install bitsandbytes==0.48.1 --break-system-packages
RUN pip3 install flash-attn==2.8.3 --no-build-isolation --break-system-packages
RUN pip3 install peft==0.17.1 --break-system-packages
RUN pip3 install xformers==0.0.32.post2 --break-system-packages


RUN pip3 install ir-measures==0.4.1 --break-system-packages
RUN pip3 install rank-bm25==0.2.2 --break-system-packages
RUN pip3 install gliner==0.2.22 --break-system-packages
RUN pip3 install rerankers[all]==0.10.0 --break-system-packages

RUN pip3 install nltk==3.9.2 --break-system-packages
RUN pip3 install sentence-transformers==5.1.1 --break-system-packages

RUN pip3 install pypdf==6.1.1 --break-system-packages
RUN pip3 install openparse==0.7.0 --break-system-packages

RUN pip3 install fastembed==0.7.3 --break-system-packages
RUN pip3 install flashrank==0.2.10 --break-system-packages

RUN pip3 install tf-keras==2.20.1 --break-system-packages

RUN pip3 install langchain==0.3.27 --break-system-packages
RUN pip3 install langchain-community==0.3.30 --break-system-packages
RUN pip3 install langchain-experimental==0.3.4 --break-system-packages
RUN pip3 install langchain-huggingface==0.3.1 --break-system-packages

RUN pip3 install dspy==3.0.3 --break-system-packages

RUN pip3 install atomic-agents==2.2.0 --break-system-packages

RUN pip3 install pylance==0.38.1 --break-system-packages
RUN pip3 install lancedb==0.25.1 --break-system-packages

RUN pip3 install huggingface-hub==0.35.3 --break-system-packages

RUN pip3 install mangoCR==0.1.4 --break-system-packages

RUN pip3 install ollm==0.5.0 --break-system-packages
RUN pip3 install ollmcp==0.18.2 --break-system-packages
RUN pip3 install deepeval==3.6.4 --break-system-packages
RUN pip3 install markitdown[all]==0.1.3 --break-system-packages
RUN pip3 install olmocr==0.4.4 --break-system-packages
RUN pip3 install easyocr==1.7.2 --break-system-packages
RUN pip3 install layoutparser==0.3.4 --break-system-packages
RUN pip3 install paddleocr==3.3.1 --break-system-packages
RUN pip3 install pdf2image==1.16.3 --break-system-packages
RUN pip3 install paddlepaddle-gpu==2.6.2 --break-system-packages

RUN pip3 install litellm==1.79.1 --break-system-packages
RUN pip3 install pillow==12.0.0 --break-system-packages
RUN pip3 install pillow-heif==1.1.1 --break-system-packages

EXPOSE 8888
WORKDIR /jupyterlab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir='/jupyterlab/notebooks'", "--allow-root"]
