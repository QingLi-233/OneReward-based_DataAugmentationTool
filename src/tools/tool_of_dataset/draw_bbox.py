import argparse
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
"""
绘制 XML 文件中的边界框
python draw_bbox.py --image ./output/fixed-224-dynamic/1766374542.3269982-13565310/output_filled.png --xml ./output/fixed-224-dynamic/1766374542.3269982-13565310/1766374542.3269982.xml --output draw_bbox7.png --color red --line_width 3

"""
def draw_bboxes(image_path, xml_path, output_path, bbox_color='red', line_width=3):
    """
    在图像上绘制 XML 中标注的边界框

    Args:
        image_path (str): 输入图像路径
        xml_path (str): 对应 VOC 格式 XML 文件路径
        output_path (str): 输出图像保存路径
        bbox_color (str): 边界框颜色，支持颜色名称或RGB元组，如 'red' 或 (255,0,0)
        line_width (int): 边界框线宽
    """
    # 1. 读取图像
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # 2. 解析 XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图像尺寸（可选，用于验证）
    size_elem = root.find('size')
    if size_elem is not None:
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        # 如果图像尺寸与 XML 不符，可给出警告
        if (width, height) != img.size:
            print(f"警告: XML 中尺寸 ({width}x{height}) 与图像实际尺寸 {img.size} 不一致")

    # 3. 遍历所有 object 并绘制边界框
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 绘制矩形框
        draw.rectangle([xmin, ymin, xmax, ymax], outline=bbox_color, width=line_width)

        # 可选：在矩形框上方添加类别名称
        try:
            # 尝试使用默认字体
            font = ImageFont.load_default()
        except:
            font = None
        draw.text((xmin, ymin - 10), name, fill=bbox_color, font=font)

    # 4. 保存图像
    img.save(output_path)
    print(f"已保存带有边界框的图像至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='在图像上绘制 VOC 格式 XML 中的边界框')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--xml', required=True, help='输入 XML 路径')
    parser.add_argument('--output', required=True, help='输出图像路径')
    parser.add_argument('--color', default='red', help='边界框颜色，如 red, blue, (255,0,0)')
    parser.add_argument('--line_width', type=int, default=3, help='边界框线宽')
    args = parser.parse_args()

    draw_bboxes(args.image, args.xml, args.output, args.color, args.line_width)

if __name__ == '__main__':
    main()