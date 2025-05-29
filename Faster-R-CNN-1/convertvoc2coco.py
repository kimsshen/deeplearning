"""
VOC格式转COCO格式转换工具
支持：目标检测标注转换（边界框）
输入：VOC格式的XML文件目录
输出：COCO格式的JSON文件
"""

import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
import argparse
import shutil
from datetime import datetime


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VOC to COCO格式转换工具')
    parser.add_argument('--voc_dir', type=str, required=True,
                        help='VOC格式数据集目录路径')
    parser.add_argument('--output', type=str, default='coco_annotations.json',
                        help='输出COCO JSON文件路径')
    parser.add_argument('--split', type=float, default=1.0,
                        help='训练集比例(0-1)，剩余为验证集')
    return parser.parse_args()


def parse_voc_annotation(xml_path):
    """解析单个VOC XML文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 解析图像基本信息
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text) if size.find('depth') is not None else 3

    # 解析所有目标对象
    objects = []
    for obj in root.iter('object'):
        # 获取类别
        cls_name = obj.find('name').text.strip().lower()

        # 跳过difficult对象
        difficult = obj.find('difficult')
        difficult = 0 if difficult is None else int(difficult.text)
        if difficult == 1:
            continue

        # 解析边界框
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 转换为COCO格式 [x_min, y_min, width, height]
        w = xmax - xmin
        h = ymax - ymin

        objects.append({
            'class': cls_name,
            'bbox': [xmin, ymin, w, h],
            'area': w * h
        })

    return {
        'filename': filename,
        'width': width,
        'height': height,
        'depth': depth,
        'objects': objects
    }


def convert_voc_to_coco(voc_dir, output_path, split_ratio=1.0):
    """主转换函数"""
    # 检查目录结构
    annotations_dir = os.path.join(voc_dir, 'annotations')
    images_dir = os.path.join(voc_dir, 'train')

    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"未找到Annotations目录: {annotations_dir}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"未找到JPEGImages目录: {images_dir}")

    # 获取所有XML文件
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    print(f"找到 {len(xml_files)} 个标注文件")

    # 初始化COCO数据结构
    coco_data = {
        "info": {
            "description": "Converted from VOC format",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "VOC2COCO Converter",
            "date_created": datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # 收集所有类别
    class_counter = defaultdict(int)
    for xml_file in tqdm(xml_files, desc="分析类别"):
        xml_path = os.path.join(annotations_dir, xml_file)
        data = parse_voc_annotation(xml_path)
        for obj in data['objects']:
            class_counter[obj['class']] += 1

    # 创建类别映射 (ID从1开始)
    categories = []
    for i, (cls_name, count) in enumerate(sorted(class_counter.items()), 1):
        categories.append({
            "id": i,
            "name": cls_name,
            "supercategory": "none",
            "instance_count": count
        })
    coco_data["categories"] = categories
    cls_name_to_id = {cat['name']: cat['id'] for cat in categories}

    print(f"找到 {len(categories)} 个类别:")
    for cat in categories:
        print(f"  - {cat['name']}: {cat['instance_count']}个实例")

    # 处理所有XML文件
    image_id = 1
    annotation_id = 1

    # 分割数据集 (如果需要)
    if split_ratio < 1.0:
        train_files = xml_files[:int(len(xml_files) * split_ratio)]
        val_files = xml_files[int(len(xml_files) * split_ratio):]
        print(f"数据集分割: 训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张")
    else:
        train_files = xml_files
        val_files = []

    # 添加图像和标注
    def process_files(files, dataset_type="train"):
        nonlocal image_id, annotation_id
        for xml_file in tqdm(files, desc=f"处理{dataset_type}集"):
            xml_path = os.path.join(annotations_dir, xml_file)
            data = parse_voc_annotation(xml_path)

            # 检查图像文件是否存在
            img_path = os.path.join(images_dir, data['filename'])
            if not os.path.exists(img_path):
                print(f"警告: 图像文件不存在 - {img_path}")
                continue

            # 添加图像信息
            img_info = {
                "id": image_id,
                "file_name": data['filename'],
                "width": data['width'],
                "height": data['height'],
                "license": 1,
                "coco_url": "",
                "date_captured": "",
                "flickr_url": "",
                "dataset": dataset_type
            }
            coco_data["images"].append(img_info)

            # 添加标注信息
            for obj in data['objects']:
                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls_name_to_id[obj['class']],
                    "bbox": obj['bbox'],
                    "area": obj['area'],
                    "iscrowd": 0,
                    "segmentation": []  # VOC不包含分割信息
                }
                coco_data["annotations"].append(ann)
                annotation_id += 1

            image_id += 1

    # 处理训练集和验证集
    process_files(train_files, "train")
    process_files(val_files, "val")

    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成! 保存到 {output_path}")
    print(f"统计信息:")
    print(f"  图像数量: {len(coco_data['images'])}")
    print(f"  标注数量: {len(coco_data['annotations'])}")
    print(f"  类别数量: {len(coco_data['categories'])}")



if __name__ == "__main__":

    voc_path = "data"
    output_path = "data/annotations.json"
    convert_voc_to_coco(voc_path, output_path, 1.0)