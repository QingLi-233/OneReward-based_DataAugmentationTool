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
    """基于原始XML，生成新XML，更新size和bbox坐标"""
    tree = ET.parse(original_xml_path)
    root = tree.getroot()

    size_elem = root.find('size')
    if size_elem is not None:
        width_elem = size_elem.find('width')
        height_elem = size_elem.find('height')
        if width_elem is not None:
            width_elem.text = str(target_size)
        if height_elem is not None:
            height_elem.text = str(target_size)

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is None:
            continue
        xmin_orig = int(bbox.find('xmin').text)
        ymin_orig = int(bbox.find('ymin').text)
        xmax_orig = int(bbox.find('xmax').text)
        ymax_orig = int(bbox.find('ymax').text)

        xmin_crop = xmin_orig - crop_left
        ymin_crop = ymin_orig - crop_top
        xmax_crop = xmax_orig - crop_left
        ymax_crop = ymax_orig - crop_top

        new_xmin = xmin_crop + offset_x
        new_ymin = ymin_crop + offset_y
        new_xmax = xmax_crop + offset_x
        new_ymax = ymax_crop + offset_y

        new_xmin = max(0, new_xmin)
        new_ymin = max(0, new_ymin)
        new_xmax = min(target_size, new_xmax)
        new_ymax = min(target_size, new_ymax)

        bbox.find('xmin').text = str(new_xmin)
        bbox.find('ymin').text = str(new_ymin)
        bbox.find('xmax').text = str(new_xmax)
        bbox.find('ymax').text = str(new_ymax)

    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
#加载模型
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

    pipe.enable_sequential_cpu_offload()
    
    # pipe.enable_model_cpu_offload()
    return pipe

def process_one_image(pipe, image_path, xml_path, output_dir, args, seed):
    """处理单张图像"""
    full_image = load_image(image_path)
    img_width, img_height = full_image.size

    # 解析XML，取第一个bbox
    _, _, bboxes = load_xml(xml_path)
    if not bboxes:
        print(f"  跳过：XML中未找到任何bbox")
        return
    name, xmin_orig, ymin_orig, xmax_orig, ymax_orig = bboxes[0]

    # 根据模式计算裁剪区域
    if args.crop_mode == "expand":
        if args.scale_factor != 1.0:
            cx = (xmin_orig + xmax_orig) / 2.0
            cy = (ymin_orig + ymax_orig) / 2.0
            w = (xmax_orig - xmin_orig) * args.scale_factor
            h = (ymax_orig - ymin_orig) * args.scale_factor
            xmin = int(cx - w / 2)
            ymin = int(cy - h / 2)
            xmax = int(cx + w / 2)
            ymax = int(cy + h / 2)
        else:
            xmin, ymin, xmax, ymax = xmin_orig, ymin_orig, xmax_orig, ymax_orig
    else:  # fixed模式
        cx = (xmin_orig + xmax_orig) / 2.0
        cy = (ymin_orig + ymax_orig) / 2.0
        # half_w = (xmax_orig - xmin_orig) / 2.0 + args.fixed_pixels
        # half_h = (ymax_orig - ymin_orig) / 2.0 + args.fixed_pixels
        xmin = int(cx - args.fixed_pixels)
        ymin = int(cy - args.fixed_pixels)
        xmax = int(cx + args.fixed_pixels)
        ymax = int(cy + args.fixed_pixels)

    # 确保不超出原图边界
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(img_height, ymax)

    # 从原图中裁剪出区域
    crop_img = full_image.crop((xmin, ymin, xmax, ymax))
    crop_w, crop_h = crop_img.size

    # 创建画布和mask
    canvas = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    mask = Image.new('L', (TARGET_SIZE, TARGET_SIZE), 255)

    # 随机确定放置位置（偏移量）
    random.seed(seed)
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
    canvas.paste(crop_img, (offset_x, offset_y))
    draw = ImageDraw.Draw(mask)
    draw.rectangle([offset_x, offset_y, offset_x + crop_w, offset_y + crop_h], fill=0)

    # 保存中间产物
    img_name = os.path.basename(image_path)
    shutil.copy2(image_path, os.path.join(output_dir, img_name))
    crop_img.save(os.path.join(output_dir, "crop.png"))
    canvas.save(os.path.join(output_dir, "canvas_before_fill.png"))
    mask.save(os.path.join(output_dir, "mask.png"))

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
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]

    out_img_path = os.path.join(output_dir, "output_filled.png")
    result.save(out_img_path)
    print(f"  填充结果已保存: {out_img_path}")

    # 生成新XML
    out_xml_path = os.path.join(output_dir, os.path.basename(xml_path))
    update_xml_for_new_image(xml_path, out_xml_path, xmin, ymin, offset_x, offset_y, TARGET_SIZE)
    print(f"  新XML已保存: {out_xml_path}")

    # 保存参数记录
    param_file = os.path.join(output_dir, "params.txt")
    with open(param_file, 'w') as f:
        f.write("=== 运行参数 ===\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n=== 生成信息 ===\n")
        f.write(f"seed_used: {seed}\n")
        f.write(f"original_image_size: {img_width}x{img_height}\n")
        f.write(f"crop_bbox (original): {xmin_orig}, {ymin_orig}, {xmax_orig}, {ymax_orig}\n")
        f.write(f"crop_mode: {args.crop_mode}\n")
        if args.crop_mode == "expand":
            f.write(f"scale_factor: {args.scale_factor}\n")
        else:
            f.write(f"fixed_pixels: {args.fixed_pixels}\n")
        f.write(f"final_crop_bbox: {xmin}, {ymin}, {xmax}, {ymax}\n")
        f.write(f"crop_size: {crop_w}x{crop_h}\n")
        f.write(f"offset_on_canvas: {offset_x}, {offset_y}\n")
        f.write(f"output_dir: {output_dir}\n")
    print(f"  参数已记录: {param_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="输入图像目录")
    parser.add_argument("--xml_root", required=True, help="XML文件存放根目录")
    parser.add_argument("--output_root", required=True, help="输出根目录")
    parser.add_argument("--crop_mode", type=str, default="expand", choices=["expand", "fixed"], 
                        help="裁剪模式: expand(扩大bbox) 或 fixed(以bbox为中心固定扩展)")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="expand模式下，bbox扩大倍率")
    parser.add_argument("--fixed_pixels", type=int, default=100, help="fixed模式下，向bbox四周扩展的像素数")
    parser.add_argument("--shift_range", type=int, default=0, help="放置裁剪图时的随机偏移范围（像素），0表示居中")
    parser.add_argument("--prompt", type=str, default="high-definition, perfect composition", help="填充提示词")
    parser.add_argument("--negative_prompt", type=str, default="nsfw", help="负向提示词")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="guidance_scale")
    parser.add_argument("--true_cfg", type=float, default=4.0, help="true_cfg")
    parser.add_argument("--model_type", type=str, default="dynamic", choices=["onereward", "dynamic"])
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise NotADirectoryError(f"图像目录不存在: {args.image_dir}")
    if not os.path.isdir(args.xml_root):
        raise NotADirectoryError(f"XML根目录不存在: {args.xml_root}")

    os.makedirs(args.output_root, exist_ok=True)

    # 保存本次运行的全局参数
    global_param_file = os.path.join(args.output_root, "run_params.txt")
    with open(global_param_file, 'w') as f:
        f.write("=== Run Parameters ===\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"全局参数已保存到: {global_param_file}")

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    img_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(img_extensions)]
    if not img_files:
        print("未找到任何图像文件")
        return

    print("正在加载模型...")
    pipe = build_pipeline(args.model_type)
    print("模型加载完成")

    for idx, img_file in enumerate(img_files, 1):
        print(f"\n处理第 {idx}/{len(img_files)} 张: {img_file}")
        seed = random.randint(0, 2**31 - 1)
        print(f"  随机种子: {seed}")

        img_stem = os.path.splitext(img_file)[0]
        subdir_name = f"{img_stem}-{seed}"
        output_dir = os.path.join(args.output_root, subdir_name)
        os.makedirs(output_dir, exist_ok=True)

        xml_path = os.path.join(args.xml_root, img_stem + ".xml")
        if not os.path.exists(xml_path):
            print(f"  跳过：XML文件不存在 {xml_path}")
            continue

        image_path = os.path.join(args.image_dir, img_file)
        process_one_image(pipe, image_path, xml_path, output_dir, args, seed)

    print("\n全部处理完成")

if __name__ == "__main__":
    main()

"""
根据bbox扩展进行裁剪
python /home/jrx/cygy/competition/scripts/oneReward/better_run_betch.py \
    --image_dir /home/jrx/cygy/competition/data/office/images \
    --xml_root /home/jrx/cygy/competition/data/office/xmls \
    --output_root /home/jrx/cygy/output/expand-2-4-dynamic \
    --crop_mode expand \
    --scale_factor 2.0 \
    --shift_range 256 \
    --prompt "high-definition, perfect composition, metal texture, power transmission line hardware, insulator string connectors, metal clevis and clevis pins, galvanized steel fittings on electrical tower, industrial electrical infrastructure, close-up view of transmission tower components, overhead power line fittings" \
    --steps 40 \
    --negative_prompt "low-quality, bad composition, nsfw, text, cartoon, extra bolt and nut"

"""

"""
根据bbox中心固定裁剪正方形图像
python /home/jrx/cygy/competition/scripts/oneReward/better_run_betch.py \
    --image_dir /home/jrx/cygy/competition/data/office/images \
    --xml_root /home/jrx/cygy/competition/data/office/xmls \
    --output_root /home/jrx/cygy/output/fixed-448-dynamic \
    --crop_mode fixed \
    --fixed_pixels 448 \
    --shift_range 150 \
    --prompt "high-definition, perfect composition, metal texture, power transmission line hardware, metal clevis and clevis pins, galvanized steel fittings on electrical tower, industrial electrical infrastructure, close-up view of steel fittings" \
    --steps 40 \
    --negative_prompt "low-quality, bad composition, nsfw, text, cartoon, bolt, nut"
"""