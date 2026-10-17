import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# 支持的图像扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def extract_number(filename: str) -> int:
    """从文件名 'defect_bolt_数字.扩展名' 中提取数字部分"""
    match = re.search(r'defect_bolt_(\d+)', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"文件名 '{filename}' 不符合 'defect_bolt_编号' 格式")

def find_paired_files(image_dir: Path, xml_dir: Path) -> List[Tuple[Path, Path, int]]:
    """
    找出图像目录和 XML 目录中配对的（同名且编号相同）文件对。
    返回列表，每项为 (image_path, xml_path, 原始编号)
    """
    # 收集所有图像文件
    images = {}
    for ext in IMAGE_EXTENSIONS:
        for img_path in image_dir.glob(f"defect_bolt_*{ext}"):
            try:
                num = extract_number(img_path.name)
                images[num] = img_path
            except ValueError:
                print(f"警告: 跳过不符合命名格式的图像文件 {img_path.name}")
                continue

    # 收集所有 XML 文件
    xmls = {}
    for xml_path in xml_dir.glob("defect_bolt_*.xml"):
        try:
            num = extract_number(xml_path.name)
            xmls[num] = xml_path
        except ValueError:
            print(f"警告: 跳过不符合命名格式的XML文件 {xml_path.name}")
            continue

    # 找出共同存在的编号
    common_nums = set(images.keys()) & set(xmls.keys())
    if not common_nums:
        print("错误: 没有找到任何配对的图像和XML文件。")
        sys.exit(1)

    # 按原始编号排序
    paired = []
    for num in sorted(common_nums):
        paired.append((images[num], xmls[num], num))

    # 报告缺失情况
    missing_img = set(xmls.keys()) - set(images.keys())
    missing_xml = set(images.keys()) - set(xmls.keys())
    if missing_img:
        print(f"警告: 以下编号有XML但无对应图像: {sorted(missing_img)}")
    if missing_xml:
        print(f"警告: 以下编号有图像但无对应XML: {sorted(missing_xml)}")

    return paired

def reorder_files(image_dir: str, xml_dir: str, start_num: int = 1, dry_run: bool = False):
    """
    将图像和XML文件重新排序，从 start_num 开始连续编号。
    dry_run=True 时仅预览操作，不实际重命名。
    """
    image_path = Path(image_dir)
    xml_path = Path(xml_dir)

    if not image_path.is_dir():
        print(f"错误: 图像目录 '{image_dir}' 不存在。")
        sys.exit(1)
    if not xml_path.is_dir():
        print(f"错误: XML目录 '{xml_dir}' 不存在。")
        sys.exit(1)

    paired = find_paired_files(image_path, xml_path)
    if not paired:
        print("没有找到可配对的图像和XML文件，操作终止。")
        return

    print(f"找到 {len(paired)} 组文件，将重新编号为 {start_num} 至 {start_num + len(paired) - 1}")

    # 生成重命名映射，避免冲突
    rename_ops = []  # (原路径, 新路径)
    new_nums = list(range(start_num, start_num + len(paired)))

    # 先检查目标文件是否已存在（但排除将要被重命名的原文件本身）
    # 由于新编号很可能与某些旧编号重叠，需要小心处理。最简单的方法是先全部改成临时名，
    # 或者采用两阶段重命名。这里采用临时文件夹方式以避免冲突。
    # 方案：先创建一个临时目录，将文件移动到临时目录并改名，再移回来。
    # 但对于简单的编号递增且起始编号可能大于现有最大编号的情况，冲突概率较低。
    # 但为了安全，我们采用“先收集映射，检测冲突，有冲突则提示是否继续”策略。
    conflicts = []
    for (img_old, xml_old, old_num), new_num in zip(paired, new_nums):
        img_new = img_old.parent / f"defect_bolt_{new_num}{img_old.suffix}"
        xml_new = xml_old.parent / f"defect_bolt_{new_num}{xml_old.suffix}"
        if img_new.exists() and img_new != img_old:
            conflicts.append(f"图像目标已存在: {img_new}")
        if xml_new.exists() and xml_new != xml_old:
            conflicts.append(f"XML目标已存在: {xml_new}")
        rename_ops.append((img_old, img_new))
        rename_ops.append((xml_old, xml_new))

    if conflicts:
        print("检测到以下冲突，无法安全重命名:")
        for c in conflicts:
            print(f"  {c}")
        response = input("是否仍要继续？目标文件将被覆盖 (y/N): ").strip().lower()
        if response != 'y':
            print("操作已取消。")
            return
        # 若用户确认覆盖，则继续执行（已有映射）

    if dry_run:
        print("\n[DRY RUN] 将执行以下重命名操作:")
        for old, new in rename_ops:
            print(f"  {old} -> {new}")
        return

    # 实际执行重命名
    print("\n开始重命名...")
    for old, new in rename_ops:
        try:
            old.rename(new)
            print(f"已重命名: {old.name} -> {new.name}")
        except Exception as e:
            print(f"重命名失败: {old} -> {new}, 错误: {e}")
            sys.exit(1)

    print("重命名完成！")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="将图像和对应的XML文件按编号重新排序")
    parser.add_argument("image_dir", default="D:\桌面文件\编程\比赛\最终数据集\DataFiles", help="图像所在目录")
    parser.add_argument("xml_dir", default="D:\桌面文件\编程\比赛\最终数据集\Annotations", help="XML文件所在目录")
    parser.add_argument("--start", type=int, default=341, help="起始编号 (默认: 1)")
    parser.add_argument("--dry-run", action="store_true", help="仅预览操作，不实际重命名")
    args = parser.parse_args()

    reorder_files(args.image_dir, args.xml_dir, args.start, args.dry_run)

if __name__ == "__main__":
    main()