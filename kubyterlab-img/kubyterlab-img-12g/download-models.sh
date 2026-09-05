#!/usr/bin/env bash
# Populates /home/sinan/hf/hub and /home/sinan/models from scratch, on a machine
# that has no pre-built kubyterlab-img-12g image to copy from.
#
# Strategy: rather than hand-picking which files each model needs (the mistake
# that caused the Z-Image-Turbo transformer/config.json to go missing), this runs
# the SAME model-loading code the test task runs, inside the actual base image
# (so dependency versions match exactly), with real internet access and HF_HOME
# pointed at the host cache dir. Whatever huggingface_hub/diffusers/transformers
# decide they need gets cached automatically. Requires the base image to already
# exist locally or be pullable: sinanozel/kubyterlab-img:26.07
set -euo pipefail

HOST_HF=/home/sinan/hf
HOST_MODELS=/home/sinan/models
mkdir -p "$HOST_HF/hub" "$HOST_MODELS/antelopev2" "$HOST_MODELS/InstantID/ControlNetModel"

# --- Models exercised by test-kubyterlab-img-12g: load them for real, online ---
docker run --rm --gpus=all \
  -v "$HOST_HF":/jupyterlab/hf \
  -e HF_HOME=/jupyterlab/hf \
  sinanozel/kubyterlab-img:26.07 \
  python3 -c "
import torch
from diffusers import (
    ZImageTransformer2DModel, ZImagePipeline, GGUFQuantizationConfig,
    AutoPipelineForInpainting, AutoPipelineForText2Image,
)
from transformers import Qwen3Model, BitsAndBytesConfig

print('--- Z-Image-Turbo (GGUF transformer + Tongyi-MAI base) ---')
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder = Qwen3Model.from_pretrained('Tongyi-MAI/Z-Image-Turbo', subfolder='text_encoder', quantization_config=bnb_config, dtype=torch.bfloat16)
import huggingface_hub
gguf_path = huggingface_hub.hf_hub_download('jayn7/Z-Image-Turbo-GGUF', 'z_image_turbo-Q4_K_M.gguf')
transformer = ZImageTransformer2DModel.from_single_file(gguf_path, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16))
pipe = ZImagePipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo', text_encoder=text_encoder, transformer=transformer, torch_dtype=torch.bfloat16).to('cuda')
pipe('three cute cats playing', width=1024, height=1024, num_inference_steps=8, guidance_scale=0.0)

print('--- Fluently v4 Inpainting ---')
AutoPipelineForInpainting.from_pretrained('fluently/Fluently-v4-inpainting', torch_dtype=torch.float16, safety_checker=None).to('cuda')

print('--- Dreamshaper 8 Inpainting ---')
AutoPipelineForInpainting.from_pretrained('Lykon/dreamshaper-8-inpainting', torch_dtype=torch.float16, safety_checker=None).to('cuda')

print('--- Realistic Vision V6 ---')
pipe = AutoPipelineForText2Image.from_pretrained('SG161222/Realistic_Vision_V6.0_B1_noVAE', torch_dtype=torch.float16, safety_checker=None).to('cuda')
pipe('a professional portrait photo', width=512, height=512, num_inference_steps=5)

print('--- Dreamshaper XL v2 Turbo ---')
pipe = AutoPipelineForText2Image.from_pretrained('Lykon/dreamshaper-xl-v2-turbo', torch_dtype=torch.float16, variant='fp16').to('cuda')
pipe('three cute cats playing', width=1024, height=1024, num_inference_steps=5)

print('--- Stable Diffusion XL Base 1.0 ---')
pipe = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, variant='fp16').to('cuda')
pipe('three cute cats playing', width=1024, height=1024, num_inference_steps=5)

print('All models loaded and cached successfully.')
"

# --- Models only COPY'd into the image (no exercising code in the test task) ---
# Full snapshot download is the right call here since there's no narrower usage to mirror.
docker run --rm -v "$HOST_HF":/jupyterlab/hf -e HF_HOME=/jupyterlab/hf sinanozel/kubyterlab-img:26.07 \
  huggingface-cli download diffusers/stable-diffusion-xl-1.0-inpainting-0.1
docker run --rm -v "$HOST_HF":/jupyterlab/hf -e HF_HOME=/jupyterlab/hf sinanozel/kubyterlab-img:26.07 \
  huggingface-cli download destitech/controlnet-inpaint-dreamer-sdxl
docker run --rm -v "$HOST_HF":/jupyterlab/hf -e HF_HOME=/jupyterlab/hf sinanozel/kubyterlab-img:26.07 \
  huggingface-cli download wangqixun/YamerMIX_v8

# --- Manually-downloaded models (not on the standard HF hub cache layout) ---
# antelopev2: the original deepinsight/insightface GitHub release (v0.7/antelopev2.zip)
# is dead; this HF mirror re-uploads the identical zip contents.
curl -fsSL "https://huggingface.co/vladmandic/insightface-faceanalysis/resolve/main/antelopev2.zip" -o /tmp/antelopev2.zip
unzip -oq /tmp/antelopev2.zip -d "$HOST_MODELS/antelopev2"
rm /tmp/antelopev2.zip

# InstantID: official InstantX/InstantID repo
curl -fsSL "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin" -o "$HOST_MODELS/InstantID/ip-adapter.bin"
curl -fsSL "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/config.json" -o "$HOST_MODELS/InstantID/ControlNetModel/config.json"
curl -fsSL "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors" -o "$HOST_MODELS/InstantID/ControlNetModel/diffusion_pytorch_model.safetensors"

echo "Done. Populated $HOST_HF/hub and $HOST_MODELS"
