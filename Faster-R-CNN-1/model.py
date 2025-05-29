import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms as T
from PIL import Image
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time
from torch.optim.lr_scheduler import StepLR
import copy
from collections import defaultdict

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义工业检测中的5类物体
INDUSTRIAL_CLASSES = {
    1: "包装盒",  # Box
    2: "标签",  # Label
    3: "瓶盖",  # Cap
    4: "密封件",  # Seal
    5: "缺陷"  # Defect
}


class IndustrialCocoDataset(Dataset):
    """工业检测COCO格式数据集加载器"""

    def __init__(self, root, annotation_file, transforms=None, class_mapping=None):
        """
        初始化数据集
        :param root: 图像根目录
        :param annotation_file: COCO标注文件路径
        :param transforms: 图像变换
        :param class_mapping: 类别映射字典
        """
        self.root = root
        self.transforms = transforms
        self.class_mapping = class_mapping or {}

        # 加载COCO格式标注
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        # 创建图像ID到图像信息的映射
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}

        # 创建图像ID到标注的映射
        self.image_id_to_annots = defaultdict(list)
        for annot in self.coco_data['annotations']:
            img_id = annot['image_id']
            self.image_id_to_annots[img_id].append(annot)

        # 获取所有图像ID
        self.image_ids = list(self.image_id_to_info.keys())

        # 创建类别ID到名称的映射
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        # 创建类别ID到索引的映射
        self.cat_id_to_idx = {}
        self.idx_to_cat_id = {}

        # 只映射我们关心的5个类别
        for idx, cat_id in enumerate(sorted(INDUSTRIAL_CLASSES.keys())):
            self.cat_id_to_idx[cat_id] = idx + 1  # +1 因为0是背景
            self.idx_to_cat_id[idx + 1] = cat_id

        # 添加背景类
        self.num_classes = len(INDUSTRIAL_CLASSES) + 1

        print(f"数据集加载完成: {len(self.image_ids)}张图像, 关注{len(INDUSTRIAL_CLASSES)}个类别")
        print("类别映射:", INDUSTRIAL_CLASSES)

    def __len__(self):
        """返回数据集中的图像数量"""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """获取单个图像及其标注"""
        # 获取图像ID和信息
        img_id = self.image_ids[idx]
        img_info = self.image_id_to_info[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])

        # 加载图像
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)  # 转换为numpy数组

        # 获取该图像的所有标注
        annots = self.image_id_to_annots[img_id]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for annot in annots:
            # 只处理我们关心的5个类别
            if annot['category_id'] not in INDUSTRIAL_CLASSES:
                continue

            # 解析边界框 [x, y, width, height] -> [x_min, y_min, x_max, y_max]
            x, y, w, h = annot['bbox']
            x_min, y_min, x_max, y_max = x, y, x + w, y + h

            # 确保边界框在图像范围内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_np.shape[1], x_max)
            y_max = min(img_np.shape[0], y_max)

            # 跳过无效边界框
            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])

            # 类别标签 (0为背景)
            cat_id = annot['category_id']
            labels.append(self.cat_id_to_idx[cat_id])

            areas.append(annot['area'])
            iscrowd.append(annot['iscrowd'])

        # 如果没有检测到任何关心的物体，创建空目标
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            areas = torch.zeros(0, dtype=torch.float32)
            iscrowd = torch.zeros(0, dtype=torch.int64)
        else:
            # 转换为张量
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([img_id])

        # 创建目标字典
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }

        # 应用变换
        if self.transforms:
            img = self.transforms(img)

        return img, target


def get_transform(train=True):
    """获取训练和验证的数据转换"""
    transforms = []
    # 转换为Tensor
    transforms.append(T.ToTensor())

    if train:
        # 训练时的数据增强
        transforms.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        # 添加随机水平翻转
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    return T.Compose(transforms)


def create_model(num_classes):
    """创建Faster R-CNN模型"""
    # 使用预训练的ResNet50作为骨干网络
    backbone = torchvision.models.resnet50(pretrained=True)
    # 移除最后的全连接层和平均池化层
    modules = list(backbone.children())[:-2]
    backbone = torch.nn.Sequential(*modules)
    backbone.out_channels = 2048  # ResNet50的输出通道数

    # 锚点生成器 - 定义不同形状和尺寸的锚框
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),),  # 不同尺度的锚框
        aspect_ratios=((0.5, 1.0, 2.0),)  # 不同宽高比
    )

    # ROI对齐 - 用于从特征图中提取固定大小的特征
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],  # 使用的特征图名称
        output_size=7,  # 输出特征图大小
        sampling_ratio=2  # 采样率
    )

    # 创建Faster R-CNN模型
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=600,  # 图像最小尺寸
        max_size=1000  # 图像最大尺寸
    )

    return model


def train_model(model, train_loader, optimizer, num_epochs=10):
    """训练模型"""
    model.train()
    model.to(device)

    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    print("开始训练...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for images, targets in train_loader:
            images = [img.to(device) for img in images]

            # 准备目标 - 跳过没有目标的数据
            valid_targets = []
            valid_images = []
            for i, t in enumerate(targets):
                if len(t["boxes"]) > 0:
                    valid_targets.append({k: v.to(device) for k, v in t.items()})
                    valid_images.append(images[i])

            # 如果没有有效目标，跳过此批次
            if len(valid_images) == 0:
                continue

            # 前向传播，计算损失
            loss_dict = model(valid_images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播和优化
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)

        print(f"轮次 [{epoch + 1}/{num_epochs}], 损失: {avg_loss:.4f}, 时间: {epoch_time:.2f}秒")

    print("训练完成!")
    return model


def check_all_classes_present(predictions, class_ids, confidence_threshold=0.7):
    """
    检查所有5类物体是否都存在
    :param predictions: 模型预测结果
    :param class_ids: 需要检测的类别ID列表
    :param confidence_threshold: 置信度阈值
    :return: (bool, dict) 是否所有类都存在, 各类别的检测结果
    """
    detected_classes = set()
    class_results = {class_id: [] for class_id in class_ids}

    # 遍历所有预测
    for pred in predictions:
        # 获取类别ID和置信度
        label = pred['labels'].item()
        score = pred['scores'].item()

        # 只考虑置信度超过阈值的检测
        if score < confidence_threshold:
            continue

        # 如果检测到我们关心的类别
        if label in class_ids:
            detected_classes.add(label)
            class_results[label].append({
                'score': score,
                'box': pred['boxes'].cpu().numpy().tolist()
            })

    # 检查是否所有5类都被检测到
    all_present = len(detected_classes) == len(class_ids)

    return all_present, class_results


def visualize_detection(image, predictions, class_mapping, output_path='detection_result.jpg'):
    """
    可视化检测结果
    :param image: 原始图像 (PIL Image)
    :param predictions: 模型预测结果
    :param class_mapping: 类别映射字典
    :param output_path: 输出图像路径
    """
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 显示图像
    ax.imshow(image)

    # 绘制每个检测结果
    for pred in predictions:
        box = pred['boxes'].cpu().numpy()
        label = pred['labels'].item()
        score = pred['scores'].item()

        # 绘制边界框
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        # 添加类别标签和置信度
        class_name = class_mapping.get(label, f'Class {label}')
        text = f"{class_name}: {score:.2f}"
        ax.text(
            box[0], box[1] - 10,
            text,
            color='red', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7)
        )

        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"检测结果已保存到: {output_path}")


class IndustrialInspector:
    """工业检测器类，封装模型和检测逻辑"""

    def __init__(self, model_path=None, device=device):
        """
        初始化检测器
        :param model_path: 预训练模型路径
        :param device: 使用的设备
        """
        self.device = device
        self.class_ids = list(range(1, 6))  # 1-5类物体
        self.class_mapping = {i: INDUSTRIAL_CLASSES[i] for i in self.class_ids}
        self.confidence_threshold = 0.6  # 置信度阈值

        # 创建模型
        self.model = create_model(len(self.class_ids) + 1)

        # 加载预训练模型
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"加载预训练模型: {model_path}")

        self.model.to(device)
        self.model.eval()  # 设置为评估模式

    def train(self, train_root, train_annot, epochs=10, save_path='industrial_inspector.pth'):
        """
        训练模型
        :param train_root: 训练图像路径
        :param train_annot: 训练标注路径
        :param epochs: 训练轮数
        :param save_path: 模型保存路径
        """
        # 创建数据集和数据加载器
        train_dataset = IndustrialCocoDataset(
            root=train_root,
            annotation_file=train_annot,
            transforms=get_transform(train=True)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,  # 小批量适合大多数GPU
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x))
        )

        # 创建优化器
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )

        # 训练模型
        self.model = train_model(self.model, train_loader, optimizer, num_epochs=epochs)

        # 保存模型
        torch.save(self.model.state_dict(), save_path)
        print(f"模型已保存到: {save_path}")

    def infer_image(self, image_path, output_image_path=None, confidence_threshold=None):
        """
        对单张图像进行推理
        :param image_path: 图像路径
        :param output_image_path: 结果图像输出路径
        :param confidence_threshold: 置信度阈值
        :return: (result, class_results) 结果OK/NG, 各类别的检测结果
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # 加载图像
        img = Image.open(image_path).convert("RGB")
        orig_image = copy.deepcopy(img)

        # 预处理图像
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        # 模型推理
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # 提取预测结果
        pred = predictions[0]  # 单张图像

        # 过滤低置信度的检测
        keep = pred['scores'] > confidence_threshold
        filtered_pred = {
            'boxes': pred['boxes'][keep],
            'labels': pred['labels'][keep],
            'scores': pred['scores'][keep]
        }

        # 检查所有类别是否存在
        all_present, class_results = check_all_classes_present(
            [{'labels': l, 'scores': s, 'boxes': b}
             for l, s, b in zip(filtered_pred['labels'], filtered_pred['scores'], filtered_pred['boxes'])],
            self.class_ids,
            confidence_threshold
        )

        # 生成结果
        result = "OK" if all_present else "NG"

        # 可视化结果（如果需要）
        if output_image_path:
            visualize_detection(orig_image, [
                {'boxes': b, 'labels': l, 'scores': s}
                for l, s, b in zip(filtered_pred['labels'], filtered_pred['scores'], filtered_pred['boxes'])
            ], self.class_mapping, output_image_path)

        # 准备详细的类别结果
        detailed_results = {}
        for class_id in self.class_ids:
            class_name = self.class_mapping[class_id]
            detected = class_id in [res['label'] for res in class_results[class_id]] if class_results[
                class_id] else False
            detailed_results[class_name] = {
                'detected': detected,
                'instances': class_results[class_id]
            }

        return result, detailed_results


def main():
    """主函数"""
    # 创建工业检测器
    inspector = IndustrialInspector()

    # 训练模型（如果有数据）
    # inspector.train(
    #     train_root="path/to/train/images",
    #     train_annot="path/to/train/annotations.json",
    #     epochs=15,
    #     save_path="industrial_inspector_model.pth"
    # )

    # 使用预训练模型（示例）
    inspector = IndustrialInspector(model_path="industrial_inspector_model.pth")

    train_root = os.path.join("data/train")
    annotation_file= os.path.join("data/annotations.json")
    inspector.train(train_root,annotation_file,10)

    # 测试图像路径
    #test_image_path = "path/to/test/image.jpg"

    # 进行推理
    # result, detailed_results = inspector.infer_image(
    #     test_image_path,
    #     output_image_path="inspection_result.jpg"
    # )

    # 打印结果
    # print("\n" + "=" * 50)
    # print(f"检测结果: {result}")
    # print("=" * 50)
    #
    # print("\n详细检测结果:")
    # for class_name, info in detailed_results.items():
    #     status = "存在" if info['detected'] else "缺失"
    #     count = len(info['instances'])
    #     print(f"- {class_name}: {status} ({count}个实例)")
    #
    # print("\n" + "=" * 50)
    # print("工业检测完成!")


if __name__ == "__main__":
    main()