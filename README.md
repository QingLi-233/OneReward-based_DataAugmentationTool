# OneReward

Official implementation of **[OneReward: Unified Mask-Guided Image Generation via Multi-Task Human Preference Learning](https://arxiv.org/abs/xxxx)**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2508.21066) [![model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/bytedance-research/OneReward) [![GitHub Pages](https://img.shields.io/badge/GitHub-Project-blue?logo=github)](https://one-reward.github.io/)
<br>

<p align="center">
  <img src="assets/show.jpg" alt="assert" width="800">
</p>

## 🚀 TODO
- [x] Release arXiv paper.
- [x] Release inference code.
- [x] Release `FLUX.1-Fill-dev[OneReward]` and `FLUX.1-Fill-dev[OneRewardDynamic]` mask-guided edit checkpoints.
- [ ] Release `FLUX.1-dev[OneReward]` text-to-image checkpoints.
- [ ] Comfyui support.
- [ ] Future open-source plan.

## Introduction
We propose **OneReward**, a novel RLHF methodology for the visual domain by employing Qwen2.5-VL as a generative reward model to enhance multitask reinforcement learning, significantly improving the policy model’s generation ability across multiple subtask. Building on OneReward, we develop **Seedream 3.0 Fill**, a unified SOTA image editing model capable of effec-tively handling diverse tasks including image fill, image extend, object removal, and text rendering. It surpasses several leading commercial and open-source systems, including Ideogram, Adobe Photoshop, and FLUX Fill [Pro]. Finally, based on FLUX Fill [dev], we are thrilled to release **FLUX.1-Fill-dev-OneReward**, which outperforms closed-source FLUX Fill [Pro] in inpainting and outpainting tasks, serving as a powerful new baseline for future research in unified image editing.

<table>
  <tr>
    <td>
      <img src="assets/radius_inpaint.png" width="512">
      <p align="center"><b>Image Fill</b></p>
    </td>
    <td>
      <img src="assets/radius_outpaint_w.png" width="512">
      <p align="center"><b>Image Extend with Prompt</b></p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="assets/radius_outpaint_wo.png" width="512">
      <p align="center"><b>Image Extend without Prompt</b></p>
    </td>
    <td>
      <img src="assets/radius_eraser.png" width="512">
      <p align="center"><b>Object Removal</b></p>
    </td>
  </tr>
  <caption align="bottom" style="font-weight: bold; margin-top: 10px;">Seedream 3.0 Fill Performance Overview</caption>
</table>

## Quick Start

1. Make sure your transformers>=4.51.3 (Supporting Qwen2.5-VL)

2. Install the latest version of diffusers (>=0.35.0)
```
pip install -U diffusers
```

The following contains a code snippet illustrating how to use the model to generate images based on text prompts and input mask, support inpaint(image-fill), outpaint(image-extend), eraser(object-removal). As the model is fully trained, FluxFillCFGPipeline with cfg is needed, you can find in [pipeline_flux_fill_with_cfg.py](src/pipeline_flux_fill_with_cfg.py).

```python
import torch
from diffusers.utils import load_image
from diffusers import FluxTransformer2DModel

from src.pipeline_flux_fill_with_cfg import FluxFillCFGPipeline

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
```

<table>
  <tr>
    <td>
      <img src="assets/image.png" width="512">
      <p align="center"><b>input</b></p>
    </td>
    <td>
      <img src="assets/result_fill.jpg" width="512">
      <p align="center"><b>output</b></p>
    </td>
  </tr>
</table>

Or you can run the whole inference demo in [demo_one_reward.py](src/examples/demo_one_reward.py) and [demo_one_reward_dynamic.py](src/examples/demo_one_reward_dynamic.py)
```python
python3 -m src.examples.demo_one_reward
python3 -m src.examples.demo_one_reward_dynamic
```

## Model
### FLUX.1-Fill-dev[OneReward], trained with Alg.1 in paper
```python
transformer_onereward = FluxTransformer2DModel.from_pretrained(
    "bytedance-research/OneReward",
    subfolder="flux.1-fill-dev-OneReward-transformer",
    torch_dtype=torch.bfloat16
)

pipe = FluxFillCFGPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", 
    transformer=transformer_onereward,
    torch_dtype=torch.bfloat16).to("cuda")
```

### FLUX.1-Fill-dev[OneRewardDynamic], trained with Alg.2 in paper
```python
transformer_onereward_dynamic = FluxTransformer2DModel.from_pretrained(
    "bytedance-research/OneReward",
    subfolder="flux.1-fill-dev-OneRewardDynamic-transformer",
    torch_dtype=torch.bfloat16
)

pipe = FluxFillCFGPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", 
    transformer=transformer_onereward_dynamic,
    torch_dtype=torch.bfloat16).to("cuda")
```

## Multi-task Usage
### Image Extend with prompt
```python
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
```

### Image Extend without prompt
```python
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
```

### Object Removal
```python
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
```

### Object Removal with Lora
As the base model flux fill have undergone heavy SFT for object generation which cause it gain only 15% usability for object removal, the improvement on removal is not obvious. we release a lora for object removal separately and might be helpful for you.

```python
import torch
from diffusers.utils import load_image
from diffusers import FluxTransformer2DModel

from src.pipeline_flux_fill_with_cfg import FluxFillCFGPipeline

transformer_onereward = FluxTransformer2DModel.from_pretrained(
    "bytedance-research/OneReward",
    subfolder="flux.1-fill-dev-OneReward-transformer",
    torch_dtype=torch.bfloat16
)

pipe = FluxFillCFGPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", 
    transformer=transformer_onereward,
    torch_dtype=torch.bfloat16).to("cuda")

pipe.load_lora_weights(
    "bytedance-research/OneReward",
    subfolder="flux.1-fill-dev-object-removal-lora",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="object_removal_lora"
)
print("Loaded adapters:", pipe.get_list_adapters())  
pipe.set_adapters(["object_removal_lora"], adapter_weights=[1.0])

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
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save(f"object_removal_lora.jpg")
```


## License Agreement
Code is licensed under Apache 2.0. Model is licensed under CC BY NC 4.0.

## Citation
```
@article{gong2025onereward,
  title={OneReward: Unified Mask-Guided Image Generation via Multi-Task Human Preference Learning},
  author={Gong, Yuan and Wang, Xionghui and Wu, Jie and Wang, Shiyin and Wang, Yitong and Wu, Xinglong},
  journal={arXiv preprint arXiv:2508.21066},
  year={2025}
}
```







