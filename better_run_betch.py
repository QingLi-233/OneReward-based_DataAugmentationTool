"""
better_run_betch.py —— 批量背景改换 + 标注迁移（阶段一 / 阶段二通用）

与 better_run.py 的差异：
    1. 遍历整个图像目录，每张图自动分配随机种子，输出目录为「图像名-种子」
    2. 支持两种裁剪模式：
       - expand：按 --scale_factor 倍率以 bbox 中心向外扩展（保留目标周围更多上下文）
       - fixed ：以 bbox 中心固定扩展 --fixed_pixels 像素（正方形裁剪）
    3. 模型只加载一次，循环处理全部图像；采用 sequential_cpu_offload 压低显存峰值
    4. 根目录写一份 run_params.txt 记录本次运行的全局参数

用法一（按 bbox 倍率外扩，阶段一：电网场景）：
    python better_run_betch.py \
        --image_dir data/office/images \
        --xml_root data/office/xmls \
        --output_root outputs/stage1-expand-2 \
        --crop_mode expand --scale_factor 2.0 --shift_range 256 \
        --prompt "high-definition, perfect composition, metal texture, power transmission line hardware, insulator string connectors, metal clevis and clevis pins, galvanized steel fittings on electrical tower, industrial electrical infrastructure, close-up view of transmission tower components, overhead power line fittings" \
        --negative_prompt "low-quality, bad composition, nsfw, text, cartoon, extra bolt and nut" \
        --steps 40

用法二（以 bbox 中心固定裁剪正方形，阶段二：自然场景）：
    python better_run_betch.py \
        --image_dir outputs/stage1/images \
        --xml_root outputs/stage1/xmls \
        --output_root outputs/stage2-fixed-448 \
        --crop_mode fixed --fixed_pixels 448 --shift_range 150 \
        --prompt "high-definition, perfect composition, metal texture, power transmission line hardware, metal clevis and clevis pins, galvanized steel fittings on electrical tower, industrial electrical infrastructure, close-up view of steel fittings" \
        --negative_prompt "low-quality, bad composition, nsfw, text, cartoon, bolt, nut" \
        --steps 40
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime

import torch
from diffusers import FluxTransformer2DModel
from diffusers.utils import load_image
from PIL import Image, ImageDraw

from src.pipeline_flux_fill_with_cfg import FluxFillCFGPipeline

# 生成画布边长（正方形）
TARGET_SIZE = 1024

# OneReward 官方模型地址
ONE_REWARD_REPO = "https://github.com/bytedance/OneReward"
ONE_REWARD_HF = "bytedance-research/OneReward"   # HuggingFace 权重仓库
FLUX_BASE_HF = "black-forest-labs/FLUX.1-Fill-dev"


def load_xml(xml_path):
    """解析 Pascal VOC XML，返回原图尺寸和 bbox 列表 [(name, xmin, ymin, xmax, ymax), ...]"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    img_width = int(size_elem.find("width").text) if size_elem is not None else None
    img_height = int(size_elem.find("height").text) if size_elem is not None else None

    bboxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        bboxes.append((name, xmin, ymin, xmax, ymax))
    return img_width, img_height, bboxes


def update_xml_for_new_image(original_xml_path, output_xml_path, crop_left, crop_top,
                              offset_x, offset_y, target_size):
    """基于原始 XML 生成新 XML：把 bbox 依次平移「原图→裁剪图→新画布」两级映射。"""
    tree = ET.parse(original_xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is not None:
        for tag in ("width", "height"):
            elem = size_elem.find(tag)
            if elem is not None:
                elem.text = str(target_size)

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin_orig = int(bbox.find("xmin").text)
        ymin_orig = int(bbox.find("ymin").text)
        xmax_orig = int(bbox.find("xmax").text)
        ymax_orig = int(bbox.find("ymax").text)

        xmin_crop = xmin_orig - crop_left
        ymin_crop = ymin_orig - crop_top
        xmax_crop = xmax_orig - crop_left
        ymax_crop = ymax_orig - crop_top

        bbox.find("xmin").text = str(max(0, xmin_crop + offset_x))
        bbox.find("ymin").text = str(max(0, ymin_crop + offset_y))
        bbox.find("xmax").text = str(min(target_size, xmax_crop + offset_x))
        bbox.find("ymax").text = str(min(target_size, ymax_crop + offset_y))

    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)


def build_pipeline(model_type="onereward", one_reward_root=None, flux_root=None):
    """加载 OneReward 增强后的 FLUX Fill [dev] 管线。"""
    if model_type == "dynamic":
        subfolder = "flux.1-fill-dev-OneRewardDynamic-transformer"
    else:
        subfolder = "flux.1-fill-dev-OneReward-transformer"

    transformer = FluxTransformer2DModel.from_pretrained(
        one_reward_root, subfolder=subfolder, torch_dtype=torch.bfloat16
    )
    pipe = FluxFillCFGPipeline.from_pretrained(
        flux_root, transformer=transformer, torch_dtype=torch.bfloat16
    )

    # 逐层 offload：显存峰值更低，适合批量长跑
    pipe.enable_sequential_cpu_offload()
    # 显存充足时可换用更快的模型级 offload：
    # pipe.enable_model_cpu_offload()
    return pipe


def place_on_canvas(crop_w, crop_h, shift_range, rng):
    """在 1024×1024 画布上随机放置裁剪图，返回 (offset_x, offset_y)。"""
    center_x = (TARGET_SIZE - crop_w) // 2
    center_y = (TARGET_SIZE - crop_h) // 2
    if shift_range <= 0:
        return center_x, center_y

    low_x = max(0, center_x - shift_range)
    high_x = min(TARGET_SIZE - crop_w, center_x + shift_range)
    low_y = max(0, center_y - shift_range)
    high_y = min(TARGET_SIZE - crop_h, center_y + shift_range)

    offset_x = rng.randint(low_x, high_x) if low_x <= high_x else center_x
    offset_y = rng.randint(low_y, high_y) if low_y <= high_y else center_y
    return offset_x, offset_y


def compute_crop_bbox(args, xmin, ymin, xmax, ymax, img_width, img_height):
    """按 crop_mode 计算实际裁剪框，并夹在原图边界内。"""
    if args.crop_mode == "expand":
        if args.scale_factor == 1.0:
            return xmin, ymin, xmax, ymax
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        w, h = (xmax - xmin) * args.scale_factor, (ymax - ymin) * args.scale_factor
        xmin, ymin = int(cx - w / 2), int(cy - h / 2)
        xmax, ymax = int(cx + w / 2), int(cy + h / 2)
    else:  # fixed：以 bbox 中心为基准固定扩展 fixed_pixels 像素
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        xmin, ymin = int(cx - args.fixed_pixels), int(cy - args.fixed_pixels)
        xmax, ymax = int(cx + args.fixed_pixels), int(cy + args.fixed_pixels)

    return max(0, xmin), max(0, ymin), min(img_width, xmax), min(img_height, ymax)


def process_one_image(pipe, image_path, xml_path, output_dir, args, seed):
    """处理单张图像：裁剪 → 偏移 → 填充 → 标注迁移。"""
    full_image = load_image(image_path)
    img_width, img_height = full_image.size

    _, _, bboxes = load_xml(xml_path)
    if not bboxes:
        print("  跳过：XML中未找到任何bbox")
        return
    name, xmin_orig, ymin_orig, xmax_orig, ymax_orig = bboxes[0]

    xmin, ymin, xmax, ymax = compute_crop_bbox(
        args, xmin_orig, ymin_orig, xmax_orig, ymax_orig, img_width, img_height
    )

    crop_img = full_image.crop((xmin, ymin, xmax, ymax))
    crop_w, crop_h = crop_img.size

    canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    mask = Image.new("L", (TARGET_SIZE, TARGET_SIZE), 255)  # 255=待生成区域

    rng = random.Random(seed)
    offset_x, offset_y = place_on_canvas(crop_w, crop_h, args.shift_range, rng)

    canvas.paste(crop_img, (offset_x, offset_y))
    ImageDraw.Draw(mask).rectangle(
        [offset_x, offset_y, offset_x + crop_w, offset_y + crop_h], fill=0  # 0=保留区域
    )

    # 中间产物
    img_name = os.path.basename(image_path)
    shutil.copy2(image_path, os.path.join(output_dir, img_name))
    crop_img.save(os.path.join(output_dir, "crop.png"))
    canvas.save(os.path.join(output_dir, "canvas_before_fill.png"))
    mask.save(os.path.join(output_dir, "mask.png"))

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=canvas,
        mask_image=mask,
        height=TARGET_SIZE,
        width=TARGET_SIZE,
        guidance_scale=args.guidance_scale,
        true_cfg=args.true_cfg,
        num_inference_steps=args.steps,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]

    out_img_path = os.path.join(output_dir, "output_filled.png")
    result.save(out_img_path)
    print(f"  填充结果已保存: {out_img_path}")

    out_xml_path = os.path.join(output_dir, os.path.basename(xml_path))
    update_xml_for_new_image(xml_path, out_xml_path, xmin, ymin, offset_x, offset_y, TARGET_SIZE)
    print(f"  新XML已保存: {out_xml_path}")

    with open(os.path.join(output_dir, "params.txt"), "w", encoding="utf-8") as f:
        f.write("=== 运行参数 ===\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n=== 生成信息 ===\n")
        f.write(f"seed_used: {seed}\n")
        f.write(f"original_image_size: {img_width}x{img_height}\n")
        f.write(f"crop_bbox (original): {xmin_orig}, {ymin_orig}, {xmax_orig}, {ymax_orig}\n")
        f.write(f"crop_mode: {args.crop_mode}\n")
        f.write(f"scale_factor: {args.scale_factor}\n" if args.crop_mode == "expand"
                else f"fixed_pixels: {args.fixed_pixels}\n")
        f.write(f"final_crop_bbox: {xmin}, {ymin}, {xmax}, {ymax}\n")
        f.write(f"crop_size: {crop_w}x{crop_h}\n")
        f.write(f"offset_on_canvas: {offset_x}, {offset_y}\n")
        f.write(f"output_dir: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="输入图像目录")
    parser.add_argument("--xml_root", required=True, help="XML 文件存放根目录（Pascal VOC 格式）")
    parser.add_argument("--output_root", required=True, help="输出根目录")
    parser.add_argument("--crop_mode", type=str, default="expand", choices=["expand", "fixed"],
                        help="裁剪模式: expand(按倍率外扩 bbox) 或 fixed(以 bbox 中心固定外扩像素)")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="expand 模式下 bbox 扩大倍率")
    parser.add_argument("--fixed_pixels", type=int, default=100,
                        help="fixed 模式下向 bbox 四周扩展的像素数（得到正方形裁剪）")
    parser.add_argument("--shift_range", type=int, default=0,
                        help="放置裁剪图时的随机偏移范围（像素），0 表示居中")
    parser.add_argument("--prompt", type=str, default="high-definition, perfect composition",
                        help="填充提示词")
    parser.add_argument("--negative_prompt", type=str, default="nsfw", help="负向提示词")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="guidance_scale")
    parser.add_argument("--true_cfg", type=float, default=4.0, help="true_cfg（OneReward 专用）")
    parser.add_argument("--model_type", type=str, default="dynamic",
                        choices=["onereward", "dynamic"],
                        help="onereward=Alg.1 固定参考模型；dynamic=Alg.2 EMA 动态参考模型")
    parser.add_argument("--one_reward_root", type=str, default=ONE_REWARD_HF,
                        help=f"OneReward transformer 权重目录或 HF 仓库 id（默认 {ONE_REWARD_HF}）")
    parser.add_argument("--flux_root", type=str, default=FLUX_BASE_HF, help=f"FLUX Fill [dev] 基座模型目录或 HF 仓库 id（默认 {FLUX_BASE_HF}）")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise NotADirectoryError(f"图像目录不存在: {args.image_dir}")
    if not os.path.isdir(args.xml_root):
        raise NotADirectoryError(f"XML根目录不存在: {args.xml_root}")

    os.makedirs(args.output_root, exist_ok=True)

    # 本次运行的全局参数留痕
    with open(os.path.join(args.output_root, "run_params.txt"), "w", encoding="utf-8") as f:
        f.write("=== Run Parameters ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    img_files = [f for f in os.listdir(args.image_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
    if not img_files:
        print("未找到任何图像文件")
        return

    print("正在加载模型...")
    pipe = build_pipeline(args.model_type, args.one_reward_root, args.flux_root)
    print("模型加载完成")

    for idx, img_file in enumerate(img_files, 1):
        print(f"\n处理第 {idx}/{len(img_files)} 张: {img_file}")
        img_stem = os.path.splitext(img_file)[0]
        xml_path = os.path.join(args.xml_root, img_stem + ".xml")
        if not os.path.exists(xml_path):
            print(f"  跳过：XML文件不存在 {xml_path}")
            continue

        # 每张图独立随机种子 → 同一原图可复现出多张背景各异的增强图
        seed = random.randint(0, 2 ** 31 - 1)
        print(f"  随机种子: {seed}")

        output_dir = os.path.join(args.output_root, f"{img_stem}-{seed}")
        os.makedirs(output_dir, exist_ok=True)
        process_one_image(pipe, os.path.join(args.image_dir, img_file), xml_path, output_dir, args, seed)

    print("\n全部处理完成")


if __name__ == "__main__":
    main()
