import os
import argparse
import torch
import shutil
import random
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw

"""
用于测试随机偏移
"""

TARGET_SIZE = 1024

def random_offset(shift_range)->(int, int):
    random.seed(datetime.now())
    crop_w=128
    crop_h=256
    if shift_range > 0:
        max_offset_x = max(0, TARGET_SIZE - crop_w - shift_range)
        max_offset_y = max(0, TARGET_SIZE - crop_h - shift_range)
        if max_offset_x > 0:
            offset_x = random.randint(-shift_range, shift_range)
        else:
            offset_x = (TARGET_SIZE - crop_w) // 2
        if max_offset_y > 0:
            offset_y = random.randint(-shift_range, shift_range)
        else:
            offset_y = (TARGET_SIZE - crop_h) // 2
    else:
        offset_x = (TARGET_SIZE - crop_w) // 2
        offset_y = (TARGET_SIZE - crop_h) // 2

        # 确保偏移后图像完全在画布内
    offset_x = max(0, min(offset_x, TARGET_SIZE - crop_w))
    offset_y = max(0, min(offset_y, TARGET_SIZE - crop_h))
    return offset_x, offset_y

if __name__ == "__main__":
    x,y=random_offset(1)
    print(x,y)