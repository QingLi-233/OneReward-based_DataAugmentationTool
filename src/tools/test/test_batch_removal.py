import os
import argparse
from pathlib import Path
import torch
from diffusers.utils import load_image
from diffusers import FluxTransformer2DModel
from pipeline_flux_fill_with_cfg import FluxFillCFGPipeline


"""
批量测试oneReward的目标擦除效果
python /home/jrx/cygy/competition/scripts/oneReward/src/batch_removal.py /home/jrx/cygy/competition/data/img_mask/images /home/jrx/cygy/competition/data/img_mask/masks /home/jrx/cygy/competition/data/img_mask/outputs
"""


# 模型路径（可根据需要修改或作为参数）
TRANSFORMER_PATH = "/home/jrx/cygy/competition/models/oneReward/flux.1-fill-dev-OneRewardDynamic-transformer"
BASE_MODEL_PATH = "/home/jrx/cygy/competition/models/flux"

# 支持的图像扩展名
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def find_mask_file(image_name: str, mask_dir: Path) -> Path | None:
    """根据图像文件名（不含扩展名）在 mask_dir 中查找匹配的掩码文件"""
    stem = Path(image_name).stem
    for ext in IMAGE_EXTS:
        mask_path = mask_dir / f"{stem}_mask.png"
        if mask_path.exists():
            return mask_path
    return None

def main():
    parser = argparse.ArgumentParser(description="批量目标擦除：根据掩码擦除图像中的对象")
    parser.add_argument("image_dir", help="输入图像目录")
    parser.add_argument("mask_dir", help="输入掩码目录（掩码文件需与图像同名）")
    parser.add_argument("output_dir", help="输出结果目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--steps", type=int, default=20, help="推理步数")
    parser.add_argument("--true_cfg", type=float, default=4.0, help="true_cfg 强度")
    args = parser.parse_args()

    # 创建输出目录
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型（全局复用）
    print("Loading transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        TRANSFORMER_PATH,
        torch_dtype=torch.bfloat16
    )
    print("Loading pipeline...")
    pipe = FluxFillCFGPipeline.from_pretrained(
        BASE_MODEL_PATH,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    # pipe.enable_model_cpu_offload()   # 节省显存
    pipe.enable_sequential_cpu_offload()

    # 遍历图像目录
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    if not image_files:
        print(f"未在 {image_dir} 中找到图像文件")
        return

    processed = 0
    for img_path in image_files:
        # 查找对应掩码
        mask_path = find_mask_file(img_path.name, mask_dir)
        if mask_path is None:
            print(f"跳过 {img_path.name}：未找到匹配的掩码")
            continue

        print(f"处理 {img_path.name} ...")
        try:
            image = load_image(str(img_path))
            mask = load_image(str(mask_path))

            result = pipe(
                prompt="remove",
                negative_prompt="nsfw",
                image=image,
                mask_image=mask,
                height=image.height,
                width=image.width,
                guidance_scale=1.0,
                true_cfg=args.true_cfg,
                num_inference_steps=args.steps,
                generator=torch.Generator("cuda").manual_seed(args.seed)
            ).images[0]

            # 保存结果，文件名保持与原图相同但扩展名改为 .png（可调整）
            out_path = out_dir / f"{img_path.stem}_removed.png"
            result.save(out_path)
            processed += 1
        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {e}")

    print(f"处理完成，共处理 {processed} 张图像，结果保存在 {out_dir}")

if __name__ == "__main__":
    main()