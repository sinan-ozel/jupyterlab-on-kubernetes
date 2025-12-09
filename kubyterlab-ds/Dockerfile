# Use the official Jupyter base image
FROM jupyter/base-notebook:latest

# Set labels for the image
LABEL org.opencontainers.image.title="kubyterlab-ds"
LABEL org.opencontainers.image.description="JupyterLab environment for data science with Redis, Qdrant, ChromaDB, and LanceDB"
LABEL org.opencontainers.image.version="25.11"
LABEL org.opencontainers.image.authors="Sinan Ozel"
LABEL org.opencontainers.image.url="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-ds"
LABEL org.opencontainers.image.source="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-ds"
LABEL org.opencontainers.image.vendor="Sinan Ozel"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.base.name="jupyter/base-notebook:latest"

# Switch to root to install packages
USER root

# Install system dependencies (optional, add what you need)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user
USER $NB_UID

# Install mature, stable packages (slower update cycles)
RUN pip install --no-cache-dir \
    scikit-learn==1.7.2 \
    lxml==6.0.2 \
    openpyxl==3.1.5 \
    beautifulsoup4==4.14.3 \
    requests==2.32.5 \
    urllib3==2.6.1 \
    charset-normalizer==3.4.4 \
    PyYAML==6.0.3 \
    fqdn==1.5.1 \
    pandocfilters==1.5.1 \
    defusedxml==0.7.1 \
    pillow==12.0.0 \
    pillow-heif==1.1.1 \
    nbdime==4.0.2 \
    tqdm==4.67.1

# Install frequently updated packages (active development)
RUN pip install --no-cache-dir \
    numpy==2.3.5 \
    pandas==2.3.3 \
    matplotlib==3.10.7 \
    seaborn==0.13.2 \
    redis==7.1.0 \
    qdrant-client==1.16.1 \
    chromadb==1.3.5 \
    lancedb==0.25.3 \
    jsonschema==4.25.1 \
    jsonschema-specifications==2025.9.1 \
    json5==0.12.1 \
    pytz==2025.2 \
    webcolors==25.10.0 \
    redis-memory==0.3.2 \
    litellm==1.80.9

# Configure git and nbdime for Jupyter
RUN nbdime config-git --enable --global

# Set the working directory
WORKDIR /home/jovyan/work

# Expose Jupyter port
EXPOSE 8888

