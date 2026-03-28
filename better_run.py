import os
import argparse
import torch
import shutil
import random
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from diffusers.utils import load_image
from diffusers import FluxTransformer2DModel
from PIL import Image, ImageDraw
from src.pipeline_flux_fill_with_cfg import FluxFillCFGPipeline

"""
优化的测试脚本，不管输入图像尺寸，自动按bbox向外延申scale倍进行切割，生成相应mask图（bbox位置随机）扩成512*512的图像，并换算bbox坐标保存xml文件

python /home/jrx/cygy/competition/scripts/oneReward/better_run.py \
    --image /home/jrx/cygy/competition/data/office/images/1766374552.80307.jpg \
    --xml_root /home/jrx/cygy/competition/data/office/xmls \
    --output_root /home/jrx/cygy/competition/scripts/oneReward/better_run_output \
    --scale_factor 2.0 \
    --seed 6534511 \
    --shift_range 256  \
    --prompt "high-definition, perfect composition, metal texture, power transmission line hardware, insulator string connectors, metal clevis and clevis pins, galvanized steel fittings on electrical tower, bolts and nuts securing power line equipment, industrial electrical infrastructure, close-up view of transmission tower components, overhead power line fittings" \
    --steps 40 \
    --negative_prompt "low-quality, bad composition, nsfw, text, cartoon" \
    --model_type dynamic
"""

TARGET_SIZE = 1024

def load_xml(xml_path):
    """解析XML，返回原图尺寸和bbox列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_elem = root.find('size')
    img_width = int(size_elem.find('width').text) if size_elem is not None else None
    img_height = int(size_elem.find('height').text) if size_elem is not None else None
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bboxes.append((name, xmin, ymin, xmax, ymax))
    return img_width, img_height, bboxes

def update_xml_for_new_image(original_xml_path, output_xml_path, crop_left, crop_top, offset_x, offset_y, target_size):
    """
    基于原始XML，生成新XML，更新size和bbox坐标。
    参数：
        original_xml_path: 原始XML路径
        output_xml_path:   输出XML路径
        crop_left, crop_top: 裁剪框左上角在原图中的坐标（即原图中裁剪区域的起点）
        offset_x, offset_y:  裁剪后的图像在新画布上的左上角偏移量
        target_size:         新画布的边长（正方形）
    """
    tree = ET.parse(original_xml_path)
    root = tree.getroot()

    # 修改size为TARGET_SIZE×TARGET_SIZE
    size_elem = root.find('size')
    if size_elem is not None:
        width_elem = size_elem.find('width')
        height_elem = size_elem.find('height')
        if width_elem is not None:
            width_elem.text = str(target_size)
        if height_elem is not None:
            height_elem.text = str(target_size)

    # 更新每个object的bbox
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is None:
            continue
        xmin_orig = int(bbox.find('xmin').text)
        ymin_orig = int(bbox.find('ymin').text)
        xmax_orig = int(bbox.find('xmax').text)
        ymax_orig = int(bbox.find('ymax').text)

        # 映射到裁剪图中的坐标
        xmin_crop = xmin_orig - crop_left
        ymin_crop = ymin_orig - crop_top
        xmax_crop = xmax_orig - crop_left
        ymax_crop = ymax_orig - crop_top

        # 再映射到新画布上的坐标
        new_xmin = xmin_crop + offset_x
        new_ymin = ymin_crop + offset_y
        new_xmax = xmax_crop + offset_x
        new_ymax = ymax_crop + offset_y

        # 确保坐标在新画布边界内（裁剪）
        new_xmin = max(0, new_xmin)
        new_ymin = max(0, new_ymin)
        new_xmax = min(target_size, new_xmax)
        new_ymax = min(target_size, new_ymax)

        bbox.find('xmin').text = str(new_xmin)
        bbox.find('ymin').text = str(new_ymin)
        bbox.find('xmax').text = str(new_xmax)
        bbox.find('ymax').text = str(new_ymax)

    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)

def build_pipeline(model_type="onereward"):
    if model_type == "dynamic":
        subfolder = "flux.1-fill-dev-OneRewardDynamic-transformer"
    else:
        subfolder = "flux.1-fill-dev-OneReward-transformer"

    transformer = FluxTransformer2DModel.from_pretrained(
        "/home/jrx/cygy/competition/models/oneReward",
        subfolder=subfolder,
        torch_dtype=torch.bfloat16
    )

    pipe = FluxFillCFGPipeline.from_pretrained(
        "/home/jrx/cygy/competition/models/flux",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 根据显存情况选择offload策略
    pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    return pipe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入大图路径")
    parser.add_argument("--xml_root", required=True, help="XML文件存放根目录")
    parser.add_argument("--output_root", required=True, help="输出根目录")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="裁剪bbox扩大倍率（以中心扩大）")
    parser.add_argument("--shift_range", type=int, default=0, help="放置裁剪图时的随机偏移范围（像素），0表示居中")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--prompt", type=str, default="high-definition, perfect composition", help="填充提示词")
    parser.add_argument("--negative_prompt", type=str, default="nsfw", help="负向提示词")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="guidance_scale")
    parser.add_argument("--true_cfg", type=float, default=4.0, help="true_cfg")
    parser.add_argument("--model_type", type=str, default="onereward", choices=["onereward", "dynamic"])
    args = parser.parse_args()

    # 根据图像文件名找到对应XML
    img_name = os.path.basename(args.image)
    img_stem = os.path.splitext(img_name)[0]
    xml_path = os.path.join(args.xml_root, img_stem + ".xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML文件不存在: {xml_path}")

    # 创建输出子目录（时间戳_种子）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = f"{timestamp}_seed{args.seed}"
    output_dir = os.path.join(args.output_root, subdir)
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print("正在加载模型...")
    pipe = build_pipeline(args.model_type)
    print("模型加载完成")

    # 加载原图
    full_image = load_image(args.image)
    img_width, img_height = full_image.size

    # 解析XML，取第一个bbox
    _, _, bboxes = load_xml(xml_path)
    if not bboxes:
        raise ValueError("XML中未找到任何bbox")
    name, xmin, ymin, xmax, ymax = bboxes[0]

    # 可选：扩大bbox（以中心扩大）
    if args.scale_factor != 1.0:
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = (xmax - xmin) * args.scale_factor
        h = (ymax - ymin) * args.scale_factor
        xmin = int(cx - w / 2)
        ymin = int(cy - h / 2)
        xmax = int(cx + w / 2)
        ymax = int(cy + h / 2)
        # 确保不超出原图边界
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_width, xmax)
        ymax = min(img_height, ymax)

    # 从原图中裁剪出bbox区域
    crop_img = full_image.crop((xmin, ymin, xmax, ymax))
    crop_w, crop_h = crop_img.size

    # 创建512×512画布（黑色背景）
    canvas = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    # 生成mask：全填充区域（255）
    mask = Image.new('L', (TARGET_SIZE, TARGET_SIZE), 255)

    # 随机确定放置位置（偏移量）
    random.seed(args.seed)
    if args.shift_range > 0:
        # 居中位置
        center_x = (TARGET_SIZE - crop_w) // 2
        center_y = (TARGET_SIZE - crop_h) // 2
        # 允许的最小/最大偏移（保证裁剪图不超出画布）
        min_offset_x = 0
        max_offset_x = TARGET_SIZE - crop_w
        min_offset_y = 0
        max_offset_y = TARGET_SIZE - crop_h

        # 以中心为基准，向两侧扩展 shift_range，但限制在允许范围内
        low_x = max(min_offset_x, center_x - args.shift_range)
        high_x = min(max_offset_x, center_x + args.shift_range)
        low_y = max(min_offset_y, center_y - args.shift_range)
        high_y = min(max_offset_y, center_y + args.shift_range)

        if low_x <= high_x:
            offset_x = random.randint(low_x, high_x)
        else:
            offset_x = center_x
        if low_y <= high_y:
            offset_y = random.randint(low_y, high_y)
        else:
            offset_y = center_y
    else:
        offset_x = (TARGET_SIZE - crop_w) // 2
        offset_y = (TARGET_SIZE - crop_h) // 2

    # 将裁剪图贴到画布上
    canvas.paste(crop_img, (offset_x, offset_y))
    # 在mask上，将裁剪图区域设为0（保留区域）
    draw = ImageDraw.Draw(mask)
    draw.rectangle([offset_x, offset_y, offset_x + crop_w, offset_y + crop_h], fill=0)

    # 保存中间产物
    shutil.copy2(args.image, os.path.join(output_dir, img_name))          # 原图
    crop_img.save(os.path.join(output_dir, "crop.png"))                   # 裁剪图
    canvas.save(os.path.join(output_dir, "canvas_before_fill.png"))       # 填充前的画布
    mask.save(os.path.join(output_dir, "mask.png"))                       # mask图

    # 执行填充
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
        generator=torch.Generator("cpu").manual_seed(args.seed)
    ).images[0]

    out_img_path = os.path.join(output_dir, "output_filled.png")
    result.save(out_img_path)
    print(f"填充结果已保存: {out_img_path}")

    # 生成新XML（更新bbox坐标）
    out_xml_path = os.path.join(output_dir, os.path.basename(xml_path))
    update_xml_for_new_image(xml_path, out_xml_path, xmin, ymin, offset_x, offset_y, TARGET_SIZE)
    print(f"新XML已保存: {out_xml_path}")

    # 保存参数记录文件
    param_file = os.path.join(output_dir, "params.txt")
    with open(param_file, 'w') as f:
        f.write("=== 运行参数 ===\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n=== 生成信息 ===\n")
        f.write(f"original_image_size: {img_width}x{img_height}\n")
        f.write(f"crop_bbox (original): {xmin}, {ymin}, {xmax}, {ymax}\n")
        f.write(f"crop_size: {crop_w}x{crop_h}\n")
        f.write(f"offset_on_canvas: {offset_x}, {offset_y}\n")
        f.write(f"output_dir: {output_dir}\n")
    print(f"参数已记录: {param_file}")

    print("全部完成")

if __name__ == "__main__":
    main()