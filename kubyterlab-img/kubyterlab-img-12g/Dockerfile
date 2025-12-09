# Use the base kubyterlab-img image
FROM sinanozel/kubyterlab-img:25.11

# Set labels for the image
LABEL org.opencontainers.image.title="kubyterlab-img-12g"
LABEL org.opencontainers.image.description="JupyterLab image with pre-installed Stable Diffusion models (12GB variant)"
LABEL org.opencontainers.image.version="25.11"
LABEL org.opencontainers.image.authors="Sinan Ozel"
LABEL org.opencontainers.image.url="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img-12g"
LABEL org.opencontainers.image.source="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img-12g"
LABEL org.opencontainers.image.vendor="Sinan Ozel"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.documentation="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img-12g"
LABEL model.compvis.stable-diffusion-v1-4.source="https://huggingface.co/CompVis/stable-diffusion-v1-4"
LABEL model.compvis.stable-diffusion-v1-4.citation="@InProceedings{Rombach_2022_CVPR, author = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Björn}, title = {High-Resolution Image Synthesis With Latent Diffusion Models}, booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, month = {June}, year = {2022}, pages = {10684-10695}}"
LABEL model.fluently.v3-inpainting.source="https://huggingface.co/fluently/Fluently-v3-inpainting"
LABEL model.fluently.v3-inpainting.license="CreativeML Open RAIL-M"

# Create the HuggingFace hub directory structure
USER root
RUN mkdir -p /jupyterlab/hf/hub

# Copy model files from host (as root to avoid permission issues)
COPY /home/sinan/hf/hub/models--CompVis--stable-diffusion-v1-4 /jupyterlab/hf/hub/models--CompVis--stable-diffusion-v1-4
COPY /home/sinan/hf/hub/models--fluently--Fluently-v3-inpainting /jupyterlab/hf/hub/models--fluently--Fluently-v3-inpainting

# Set proper ownership for the jovyan user (1000:100 are standard Jupyter notebook user/group IDs)
RUN chown -R 1000:100 /jupyterlab/hf

# Create jovyan user and home directory
RUN useradd -u 1000 -g 100 -d /home/jovyan -m jovyan && \
    mkdir -p /home/jovyan/.local/share/jupyter/runtime && \
    mkdir -p /home/jovyan/.jupyter && \
    chown -R 1000:100 /home/jovyan

# Switch back to jovyan user
USER 1000

# Set HOME environment variable for jovyan user
ENV HOME=/home/jovyan

# Install additional packages needed for Stable Diffusion
# Note: These packages are inherited from the base image
# RUN pip install --no-cache-dir \
#     diffusers \
#     transformers \
#     accelerate \
#     safetensors \
#     xformers \
#     numpy==2.1.3 \
#     pandas==2.2.3 \
#     matplotlib==3.9.2 \
#     seaborn==0.13.2

# Set environment variable for HuggingFace cache
ENV HF_HOME=/jupyterlab/hf

# Copy sample notebooks into the image
COPY --chown=1000:100 home/sinan/github/jupyterlab-on-kubernetes/kubyterlab-img/kubyterlab-img-12g/notebooks /jupyterlab/notebooks

# Set working directory
WORKDIR /jupyterlab

# Expose JupyterLab port
EXPOSE 8888