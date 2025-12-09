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

# Install Python packages with pinned versions for reproducibility
RUN pip install --no-cache-dir \
    numpy==2.1.3 \
    pandas==2.2.3 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    scikit-learn==1.5.2 \
    nbdime==4.0.2 \
    redis==6.1.0 \
    redis-memory==0.3.2 \
    qdrant-client==1.12.1 \
    chromadb==0.5.23 \
    lancedb==0.17.0 \
    lxml==5.3.0 \
    openpyxl==3.1.5 \
    tqdm==4.67.1 \
    beautifulsoup4==4.12.3 \
    requests==2.32.3 \
    urllib3==2.3.0 \
    charset-normalizer==3.4.0 \
    jsonschema==4.23.0 \
    jsonschema-specifications==2024.10.1 \
    json5==0.10.0 \
    PyYAML==6.0.2 \
    pytz==2024.2 \
    fqdn==1.5.1 \
    webcolors==24.11.1 \
    pandocfilters==1.5.1 \
    defusedxml==0.7.1 \
    pillow==12.0.0 \
    pillow-heif==1.1.1 \
    litellm==1.79.1

# Configure git and nbdime for Jupyter
RUN nbdime config-git --enable --global

# Set the working directory
WORKDIR /home/jovyan/work

# Expose Jupyter port
EXPOSE 8888

