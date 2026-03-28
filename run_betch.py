import os
import argparse
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from diffusers.utils import load_image
from diffusers import FluxTransformer2DModel
from PIL import Image, ImageDraw
from src.pipeline_flux_fill_with_cfg import FluxFillCFGPipeline

TARGET_SIZE = 1024
ORIGIN_SIZE = 512
OFFSET = (TARGET_SIZE - ORIGIN_SIZE) // 2  # 256

def prepare_extend_image(image):
    """将原始图像居中放置在TARGET_SIZE×TARGET_SIZE画布上，并生成扩图掩码"""
    w, h = image.size
    canvas = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    mask = Image.new('L', (TARGET_SIZE, TARGET_SIZE), 255)
    
    left = (TARGET_SIZE - w) // 2
    top = (TARGET_SIZE - h) // 2
    canvas.paste(image, (left, top))
    
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left, top, left + w, top + h], fill=0)
    return canvas, mask

def update_xml(xml_path, output_xml_path):
    """读取原始XML，修改size和bbox坐标，保存为相同格式"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 修改size为1024×1024
    size_elem = root.find('size')
    if size_elem is not None:
        width_elem = size_elem.find('width')
        height_elem = size_elem.find('height')
        if width_elem is not None:
            width_elem.text = str(TARGET_SIZE)
        if height_elem is not None:
            height_elem.text = str(TARGET_SIZE)
        # depth保持原值不变
    
    # 更新所有object的bbox坐标（加上偏移量）
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is None:
            continue
        # 获取原坐标（基于512×512图像）
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # 加上偏移量
        bbox.find('xmin').text = str(xmin + OFFSET)
        bbox.find('ymin').text = str(ymin + OFFSET)
        bbox.find('xmax').text = str(xmax + OFFSET)
        bbox.find('ymax').text = str(ymax + OFFSET)
    
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
        "/home/jrx/cygy/competition/models/oneReward/fluxl",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    return pipe

def process_one(pipe, img_path, xml_path, output_img_dir, output_xml_dir, args):
    """处理单张图像和XML"""
    # 加载图像
    image = load_image(img_path)
    # 生成扩图画布和掩码
    extended_canvas, mask = prepare_extend_image(image)
    
    # 执行扩展
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=extended_canvas,
        mask_image=mask,
        height=TARGET_SIZE,
        width=TARGET_SIZE,
        guidance_scale=args.guidance_scale,
        true_cfg=args.true_cfg,
        num_inference_steps=args.steps,
        generator=torch.Generator("cpu").manual_seed(args.seed)
    ).images[0]
    
    # 保存扩展后的图像
    img_name = os.path.basename(img_path)
    out_img_path = os.path.join(output_img_dir, img_name)
    result.save(out_img_path)
    
    # 更新XML并保存
    xml_name = os.path.basename(xml_path)
    out_xml_path = os.path.join(output_xml_dir, xml_name)
    update_xml(xml_path, out_xml_path)
    
    print(f"处理完成: {img_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True, help="输入根目录，包含img和xml子目录")
    parser.add_argument("--output_root", type=str, required=True, help="输出根目录，将创建img和xml子目录")
    parser.add_argument("--prompt", type=str, default="high-definition, perfect composition", help="扩图提示词")
    parser.add_argument("--negative_prompt", type=str, default="nsfw", help="负向提示词")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="guidance_scale")
    parser.add_argument("--true_cfg", type=float, default=4.0, help="true_cfg")
    parser.add_argument("--model_type", type=str, default="onereward", choices=["onereward", "dynamic"])
    args = parser.parse_args()
    
    # 创建输出目录
    output_img_dir = os.path.join(args.output_root, "img")
    output_xml_dir = os.path.join(args.output_root, "xml")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_xml_dir, exist_ok=True)
    
    # 加载模型
    print("正在加载模型...")
    pipe = build_pipeline(args.model_type)
    print("模型加载完成")
    
    # 输入目录
    input_img_dir = os.path.join(args.input_root, "img")
    input_xml_dir = os.path.join(args.input_root, "xml")
    
    # 遍历所有图像
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    img_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith(img_extensions)]
    
    for img_file in img_files:
        img_path = os.path.join(input_img_dir, img_file)
        xml_file = os.path.splitext(img_file)[0] + ".xml"
        xml_path = os.path.join(input_xml_dir, xml_file)
        
        if not os.path.exists(xml_path):
            print(f"警告: 未找到对应XML文件 {xml_path}，跳过 {img_file}")
            continue
        
        process_one(pipe, img_path, xml_path, output_img_dir, output_xml_dir, args)
    
    print("全部处理完成")

if __name__ == "__main__":
    main()