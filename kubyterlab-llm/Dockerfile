FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ARG TESSERACT_VERSION="5.5.1"
ARG TESSERACT_URL="https://api.github.com/repos/tesseract-ocr/tesseract/tarball/${TESSERACT_VERSION}"

RUN apt update
RUN apt install -y curl
RUN apt install -y nodejs
RUN apt install -y npm
RUN apt install -y python3.12
RUN apt install -y python3-pip

RUN apt install -y wget
RUN apt install -y ca-certificates
RUN apt install -y apt-transport-https
RUN apt install -y g++
RUN apt install -y make
RUN apt install -y libleptonica-dev
RUN apt install -y libtool
RUN apt install -y pkg-config

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

RUN apt install unzip zip -y
RUN apt install rsync -y

# Clean up apt
RUN rm -rf /var/cache/apk/*
RUN apt-get autoremove -y
RUN apt-get clean

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

EXPOSE 8888
WORKDIR /jupyterlab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir='/jupyterlab/notebooks'", "--allow-root"]