# Use the base kubyterlab-img image
FROM sinanozel/kubyterlab-img:26.07

# Set labels for the image
LABEL org.opencontainers.image.title="kubyterlab-img-12g"
LABEL org.opencontainers.image.description="JupyterLab image with pre-installed Stable Diffusion models (12GB variant)"
LABEL org.opencontainers.image.version="26.07"
LABEL org.opencontainers.image.authors="Sinan Ozel"
LABEL org.opencontainers.image.url="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img-12g"
LABEL org.opencontainers.image.source="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img-12g"
LABEL org.opencontainers.image.vendor="Sinan Ozel"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.documentation="https://github.com/sinan-ozel/jupyterlab-on-kubernetes/tree/main/kubyterlab-img/kubyterlab-img-12g"

# Model labels
LABEL model.fluently.v3-inpainting.source="https://huggingface.co/fluently/Fluently-v3-inpainting"
LABEL model.fluently.v3-inpainting.license="CreativeML Open RAIL-M"
LABEL model.fluently.v4-inpainting.source="https://huggingface.co/fluently/Fluently-v4-inpainting"
LABEL model.fluently.v4-inpainting.license="fluently-license"
LABEL model.lykon.dreamshaper-8-inpainting.source="https://huggingface.co/Lykon/dreamshaper-8-inpainting"
LABEL model.sg161222.realistic-vision-v6.source="https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE"
LABEL model.lykon.dreamshaper-xl-v2-turbo.source="https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo"
LABEL model.stabilityai.sdxl-base-1.0.source="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
LABEL model.tongyi-mai.z-image-turbo.source="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo"
LABEL model.jayn7.z-image-turbo-gguf.source="https://huggingface.co/jayn7/Z-Image-Turbo-GGUF"
LABEL model.diffusers.sdxl-1.0-inpainting.source="https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
LABEL model.destitech.controlnet-inpaint-dreamer-sdxl.source="https://huggingface.co/destitech/controlnet-inpaint-dreamer-sdxl"
LABEL model.wangqixun.yamermix-v8.source="https://huggingface.co/wangqixun/YamerMIX_v8"
LABEL model.insightface.antelopev2.source="https://github.com/deepinsight/insightface"
LABEL model.instantx.instantid.source="https://huggingface.co/InstantX/InstantID"

# Create the HuggingFace hub directory structure
RUN mkdir -p /jupyterlab/hf/hub

# Copy model files from host
COPY  /home/sinan/hf/hub/models--fluently--Fluently-v4-inpainting \
     /jupyterlab/hf/hub/models--fluently--Fluently-v4-inpainting
COPY  /home/sinan/hf/hub/models--Lykon--dreamshaper-8-inpainting \
     /jupyterlab/hf/hub/models--Lykon--dreamshaper-8-inpainting
COPY  /home/sinan/hf/hub/models--jayn7--Z-Image-Turbo-GGUF \
     /jupyterlab/hf/hub/models--jayn7--Z-Image-Turbo-GGUF
COPY  /home/sinan/hf/hub/models--Tongyi-MAI--Z-Image-Turbo \
     /jupyterlab/hf/hub/models--Tongyi-MAI--Z-Image-Turbo
COPY  /home/sinan/hf/hub/models--SG161222--Realistic_Vision_V6.0_B1_noVAE \
     /jupyterlab/hf/hub/models--SG161222--Realistic_Vision_V6.0_B1_noVAE
COPY  /home/sinan/hf/hub/models--Lykon--dreamshaper-xl-v2-turbo \
     /jupyterlab/hf/hub/models--Lykon--dreamshaper-xl-v2-turbo
COPY  /home/sinan/hf/hub/models--stabilityai--stable-diffusion-xl-base-1.0 \
     /jupyterlab/hf/hub/models--stabilityai--stable-diffusion-xl-base-1.0
COPY  /home/sinan/hf/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1 \
     /jupyterlab/hf/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1
COPY  /home/sinan/hf/hub/models--destitech--controlnet-inpaint-dreamer-sdxl \
     /jupyterlab/hf/hub/models--destitech--controlnet-inpaint-dreamer-sdxl
COPY  /home/sinan/hf/hub/models--wangqixun--YamerMIX_v8 \
     /jupyterlab/hf/hub/models--wangqixun--YamerMIX_v8
COPY  /home/sinan/models/antelopev2 \
     /jupyterlab/models/antelopev2
COPY  /home/sinan/models/InstantID \
     /jupyterlab/models/InstantID

# Set environment variables
ENV HF_HOME=/jupyterlab/hf
# Force huggingface_hub/transformers to read only from the local cache above and
# never attempt a network call (e.g. the HEAD request from_single_file() issues to
# fetch a base repo's config.json). Without this, a missing/incomplete local file
# triggers slow DNS-retry loops instead of an immediate, clear local-cache error.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV SHELL=/bin/bash

# Copy sample notebooks
COPY home/sinan/sinan-ozel@github.com/jupyterlab-on-kubernetes/kubyterlab-img/kubyterlab-img-12g/notebooks \
     /jupyterlab/notebooks

# Set working directory
WORKDIR /jupyterlab

# Expose JupyterLab port
EXPOSE 8888
