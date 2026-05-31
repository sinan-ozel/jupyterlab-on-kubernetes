# Use the base kubyterlab-vid image
FROM sinanozel/kubyterlab-vid:26.05

# Set labels for the image
LABEL org.opencontainers.image.title="kubyterlab-vid-12g"
LABEL org.opencontainers.image.description="JupyterLab image for AI video generation with pre-installed models (12GB variant)"
LABEL org.opencontainers.image.version="26.05"
LABEL org.opencontainers.image.authors="Sinan Ozel"
LABEL org.opencontainers.image.url="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-vid/kubyterlab-vid-12g"
LABEL org.opencontainers.image.source="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-vid/kubyterlab-vid-12g"
LABEL org.opencontainers.image.vendor="Sinan Ozel"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.documentation="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-vid/kubyterlab-vid-12g"

# Create the HuggingFace hub directory structure
RUN mkdir -p /jupyterlab/hf/hub

# Set environment variables
ENV HF_HOME=/jupyterlab/hf
ENV SHELL=/bin/bash

# Copy sample notebooks
COPY home/sinan/github/jupyterlab-on-kubernetes/kubyterlab-vid/kubyterlab-vid-12g/notebooks \
     /jupyterlab/notebooks

# Set working directory
WORKDIR /jupyterlab

# Expose JupyterLab port
EXPOSE 8888
