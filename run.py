import os
import argparse
import torch
import shutil
import xml.etree.ElementTree as ET
from diffusers.utils import load_image
from diffusers import FluxTransformer2DModel
from PIL import Image, ImageDraw
from src.pipeline_flux_fill_with_cfg import FluxFillCFGPipeline

"""
python /home/jrx/cygy/competition/scripts/oneReward/run.py \
    --image /home/jrx/cygy/competition/data/crop_512_xml/images/1766374556.6337132_00_closeup.png \
    --xml /home/jrx/cygy/competition/data/crop_512_xml/xmls/1766374556.6337132_00_closeup.xml \
    --output_dir /home/jrx/cygy/competition/scripts/oneReward/img_fill_output/3 \
    --prompt "high-definition, perfect composition" \
    --steps 50
"""

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
    
    # 更新所有object的bbox坐标（加上偏移量）
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is None:
            continue
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
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
        "/home/jrx/cygy/competition/models/flux",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    # pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    # pipe.enable_attention_slicing()

    return pipe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="原图路径（512×512）")
    parser.add_argument("--xml", type=str, required=True, help="原始XML路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录（存放原图、mask、扩展图、新XML）")
    parser.add_argument("--prompt", type=str, default="high-definition, perfect composition", help="扩图提示词")
    parser.add_argument("--negative_prompt", type=str, default="nsfw", help="负向提示词")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="guidance_scale")
    parser.add_argument("--true_cfg", type=float, default=4.0, help="true_cfg")
    parser.add_argument("--model_type", type=str, default="onereward", choices=["onereward", "dynamic"])
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("正在加载模型...")
    pipe = build_pipeline(args.model_type)
    print("模型加载完成")
    
    # 1. 复制原图到输出目录（保持文件名）
    img_filename = os.path.basename(args.image)
    out_img_path = os.path.join(args.output_dir, img_filename)
    shutil.copy2(args.image, out_img_path)
    print(f"原图已复制: {out_img_path}")
    
    # 2. 加载原图并生成mask和canvas
    image = load_image(args.image)
    canvas, mask = prepare_extend_image(image)
    
    # 保存mask图（PNG格式，灰度图）
    mask_filename = os.path.splitext(img_filename)[0] + "_mask.png"
    mask_path = os.path.join(args.output_dir, mask_filename)
    mask.save(mask_path)
    print(f"掩码已保存: {mask_path}")
    
    # 3. 执行扩展
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
    
    # 保存扩展图
    ext_filename = os.path.splitext(img_filename)[0] + "_extended.jpg"
    ext_path = os.path.join(args.output_dir, ext_filename)
    result.save(ext_path)
    print(f"扩展图已保存: {ext_path}")
    
    # 4. 更新XML并保存
    xml_filename = os.path.basename(args.xml)
    out_xml_path = os.path.join(args.output_dir, xml_filename)
    update_xml(args.xml, out_xml_path)
    print(f"新XML已保存: {out_xml_path}")
    
    print("全部完成")

if __name__ == "__main__":
    main()