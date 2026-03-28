# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from diffusers.utils import load_image
from diffusers import FluxTransformer2DModel

from ..pipeline_flux_fill_with_cfg import FluxFillCFGPipeline

transformer_onereward = FluxTransformer2DModel.from_pretrained(
    "bytedance-research/OneReward",
    subfolder="flux.1-fill-dev-OneReward-transformer",
    torch_dtype=torch.bfloat16
)

pipe = FluxFillCFGPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", 
    transformer=transformer_onereward,
    torch_dtype=torch.bfloat16).to("cuda")

# Image Fill
image = load_image('assets/image.png')
mask = load_image('assets/mask_fill.png')
image = pipe(
    prompt='the words "ByteDance", and in the next line "OneReward"',
    negative_prompt="nsfw",
    image=image,
    mask_image=mask,
    height=image.height,
    width=image.width,
    guidance_scale=1.0,
    true_cfg=4.0,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"image_fill.jpg")

# Object Removal
image = load_image('assets/image.png')
mask = load_image('assets/mask_remove.png')
image = pipe(
    prompt='remove',  # using fix prompt in object removal
    negative_prompt="nsfw",
    image=image,
    mask_image=mask,
    height=image.height,
    width=image.width,
    guidance_scale=1.0,
    true_cfg=4.0,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"object_removal.jpg")

# Image Extend with prompt
image = load_image('assets/image2.png')
mask = load_image('assets/mask_extend.png')
image = pipe(
    prompt='Deep in the forest, surronded by colorful flowers',
    negative_prompt="nsfw",
    image=image,
    mask_image=mask,
    height=image.height,
    width=image.width,
    guidance_scale=1.0,
    true_cfg=4.0,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"image_extend_w_prompt.jpg")

# Image Extend without prompt
image = load_image('assets/image2.png')
mask = load_image('assets/mask_extend.png')
image = pipe(
    prompt='high-definition, perfect composition',  # using fix prompt in image extend wo prompt
    negative_prompt="nsfw",
    image=image,
    mask_image=mask,
    height=image.height,
    width=image.width,
    guidance_scale=1.0,
    true_cfg=4.0,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save(f"image_extend_wo_prompt.jpg")