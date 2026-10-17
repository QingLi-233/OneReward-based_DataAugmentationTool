#!/usr/bin/env python3
"""
数据集整理脚本：将 defect_bolt_* 图片及其标注 XML 文件重新编号为从指定起始编号开始的连续序号。
输入目录需包含 DataFiles（图片）和 Annotations（XML）子目录。
输出目录将自动创建相同的子目录结构，并复制重命名后的文件。
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Tuple

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def find_paired_files(input_root: Path) -> List[Tuple[int, Path, Path]]:
    """
    扫描 input_root/DataFiles 和 input_root/Annotations，找出成对的图片与 XML 文件。

    返回列表，每个元素为 (original_id, image_path, xml_path)，按 original_id 升序排列。
    若某个编号只有图片或只有 XML，则跳过并打印警告。
    """
    data_dir = input_root / 'DataFiles'
    anno_dir = input_root / 'Annotations'

    if not data_dir.is_dir():
        raise NotADirectoryError(f"DataFiles 子目录不存在: {data_dir}")
    if not anno_dir.is_dir():
        raise NotADirectoryError(f"Annotations 子目录不存在: {anno_dir}")

    # 正则匹配 "defect_bolt_数字.扩展名"
    pattern = re.compile(r'^defect_bolt_(\d+)\.(\w+)$', re.IGNORECASE)

    paired = []

    # 遍历 DataFiles 下的所有文件
    for img_path in data_dir.iterdir():
        if not img_path.is_file():
            continue
        match = pattern.match(img_path.name)
        if not match:
            continue
        img_ext = img_path.suffix.lower()
        if img_ext not in IMAGE_EXTENSIONS:
            continue

        img_id = int(match.group(1))
        # 对应的 XML 文件名
        xml_name = f"defect_bolt_{img_id}.xml"
        xml_path = anno_dir / xml_name

        if xml_path.is_file():
            paired.append((img_id, img_path, xml_path))
        else:
            print(f"警告: 图片 {img_path.name} 缺少对应的 XML 文件，已跳过")

    # 检查是否有孤立的 XML 文件
    for xml_path in anno_dir.glob('defect_bolt_*.xml'):
        match = pattern.match(xml_path.name)
        if match:
            img_id = int(match.group(1))
            img_name_candidates = [f"defect_bolt_{img_id}{ext}" for ext in IMAGE_EXTENSIONS]
            if not any((data_dir / name).is_file() for name in img_name_candidates):
                print(f"警告: XML 文件 {xml_path.name} 缺少对应的图片，已跳过")

    # 按原始编号排序
    paired.sort(key=lambda x: x[0])
    return paired


def reorganize_dataset(input_root: Path, output_root: Path, start_id: int = 1, copy: bool = True):
    """
    重新编号并复制/移动文件到输出目录。

    :param start_id: 新编号的起始数字（默认为 1）
    :param copy: True 表示复制，False 表示移动（剪切）
    """
    if start_id < 1:
        raise ValueError("起始编号必须为正整数")

    paired = find_paired_files(input_root)

    if not paired:
        print("未找到任何有效的图片-XML 对，程序退出。")
        return

    print(f"找到 {len(paired)} 个有效样本，原始编号范围: {paired[0][0]} ~ {paired[-1][0]}")
    print(f"新编号将从 {start_id} 开始连续分配")

    # 创建输出目录结构
    out_data_dir = output_root / 'DataFiles'
    out_anno_dir = output_root / 'Annotations'
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_anno_dir.mkdir(parents=True, exist_ok=True)

    # 重新编号并处理文件
    for new_id, (old_id, img_path, xml_path) in enumerate(paired, start=start_id):
        # 新文件名
        img_ext = img_path.suffix
        new_img_name = f"defect_bolt_{new_id}{img_ext}"
        new_xml_name = f"defect_bolt_{new_id}.xml"

        dest_img = out_data_dir / new_img_name
        dest_xml = out_anno_dir / new_xml_name

        # 复制或移动
        try:
            if copy:
                shutil.copy2(img_path, dest_img)
                shutil.copy2(xml_path, dest_xml)
            else:
                shutil.move(str(img_path), str(dest_img))
                shutil.move(str(xml_path), str(dest_xml))
        except Exception as e:
            print(f"错误: 处理文件 {img_path.name} 时失败: {e}")
            continue

        print(f"编号 {old_id:3d} -> {new_id:3d}: {new_img_name}")

    print(f"\n整理完成！输出目录: {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="将 defect_bolt_* 图片及标注 XML 重新编号为连续序号"
    )
    parser.add_argument('input_dir', type=str, help="输入根目录（包含 DataFiles 和 Annotations 子目录）")
    parser.add_argument('output_dir', type=str, help="输出根目录（将创建 DataFiles 和 Annotations 子目录）")
    parser.add_argument('--start', type=int, default=1, help="新编号的起始数字（默认 1）")
    parser.add_argument('--move', action='store_true', help="移动文件而非复制（默认复制）")
    args = parser.parse_args()

    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    if not input_path.is_dir():
        print(f"错误: 输入目录不存在: {input_path}")
        return

    if args.start < 1:
        print("错误: 起始编号必须大于等于 1")
        return

    reorganize_dataset(input_path, output_path, start_id=args.start, copy=not args.move)


if __name__ == '__main__':
    main()

"""
python reorganize.py dataset/test dataset/yolo/test --start 1
"""