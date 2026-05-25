# Use the base kubyterlab-img image
FROM sinanozel/kubyterlab-img:26.05

# Set labels for the image
LABEL org.opencontainers.image.title="kubyterlab-img-12g"
LABEL org.opencontainers.image.description="JupyterLab image with pre-installed Stable Diffusion models (12GB variant)"
LABEL org.opencontainers.image.version="26.04"
LABEL org.opencontainers.image.authors="Sinan Ozel"
LABEL org.opencontainers.image.url="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img-12g"
LABEL org.opencontainers.image.source="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img-12g"
LABEL org.opencontainers.image.vendor="Sinan Ozel"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.documentation="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img-12g"

# Model labels
LABEL model.compvis.stable-diffusion-v1-4.source="https://huggingface.co/CompVis/stable-diffusion-v1-4"
LABEL model.compvis.stable-diffusion-v1-4.citation="@InProceedings{Rombach_2022_CVPR, author = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Björn}, title = {High-Resolution Image Synthesis With Latent Diffusion Models}, booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, month = {June}, year = {2022}, pages = {10684-10695}}"
LABEL model.fluently.v3-inpainting.source="https://huggingface.co/fluently/Fluently-v3-inpainting"
LABEL model.fluently.v3-inpainting.license="CreativeML Open RAIL-M"
LABEL model.fluently.v4-inpainting.source="https://huggingface.co/fluently/Fluently-v4-inpainting"
LABEL model.fluently.v4-inpainting.license="fluently-license"
LABEL model.lykon.dreamshaper-8-inpainting.source="https://huggingface.co/Lykon/dreamshaper-8-inpainting"
LABEL model.sg161222.realistic-vision-v6.source="https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE"
LABEL model.lykon.dreamshaper-xl-v2-turbo.source="https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo"
LABEL model.stabilityai.sdxl-base-1.0.source="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"

# Create the HuggingFace hub directory structure
RUN mkdir -p /jupyterlab/hf/hub

# Copy model files from host
COPY  /mnt/f/HuggingFaceCache/hub/models--CompVis--stable-diffusion-v1-4 \
     /jupyterlab/hf/hub/models--CompVis--stable-diffusion-v1-4
COPY  /mnt/f/HuggingFaceCache/hub/models--fluently--Fluently-v3-inpainting \
     /jupyterlab/hf/hub/models--fluently--Fluently-v3-inpainting
COPY  /mnt/f/HuggingFaceCache/hub/models--fluently--Fluently-v4-inpainting \
     /jupyterlab/hf/hub/models--fluently--Fluently-v4-inpainting
COPY  /mnt/f/HuggingFaceCache/hub/models--Lykon--dreamshaper-8-inpainting \
     /jupyterlab/hf/hub/models--Lykon--dreamshaper-8-inpainting
COPY  /mnt/f/HuggingFaceCache/hub/models--SG161222--Realistic_Vision_V6.0_B1_noVAE \
     /jupyterlab/hf/hub/models--SG161222--Realistic_Vision_V6.0_B1_noVAE
COPY  /mnt/f/HuggingFaceCache/hub/models--Lykon--dreamshaper-xl-v2-turbo \
     /jupyterlab/hf/hub/models--Lykon--dreamshaper-xl-v2-turbo
COPY  /mnt/f/HuggingFaceCache/hub/models--stabilityai--stable-diffusion-xl-base-1.0 \
     /jupyterlab/hf/hub/models--stabilityai--stable-diffusion-xl-base-1.0

# Set environment variables
ENV HF_HOME=/jupyterlab/hf
ENV SHELL=/bin/bash

# Copy sample notebooks
COPY home/sinan/github/jupyterlab-on-kubernetes/kubyterlab-img/kubyterlab-img-12g/notebooks \
     /jupyterlab/notebooks

# Set working directory
WORKDIR /jupyterlab

# Expose JupyterLab port
EXPOSE 8888
