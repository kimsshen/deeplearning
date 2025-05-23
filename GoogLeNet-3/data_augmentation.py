import cv2
import os
import albumentations as A
from tqdm import tqdm  # 进度条工具



# 定义增强管道
transform = A.Compose([
    A.HorizontalFlip(p=0.5), #以50%的概率水平翻转
    A.RandomBrightnessContrast(p=0.3), #以 30% 的概率改变图像的亮度和对比度。
    A.Rotate(limit=30, p=0.5),#以 50% 的概率将图像旋转最多 30 度
    A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)), #从原图中随机裁剪一块区域，然后将其缩放到 256x256 像素，缩放比例在 0.8 到 1.0 之间
    A.CoarseDropout(max_holes=8,  # 最大遮挡区域数量
        hole_size=(20, 40),  # 每个遮挡区域的大小范围（像素）
        fill_value=0  # 遮挡区域填充值，默认为0（黑色）
    )
])


def batch_augment_albumentations(input_dir, output_dir, augment_times=10):
    """
    批量增强目录中的所有图像
    参数:
        input_dir: 输入图像目录
        output_dir: 输出增强图像目录
        augment_times: 每张图像的增强次数
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        print(f"当前目录: {root}")
        for dir in dirs:
            print(f"  目录: {os.path.join(root, dir)}")
            output_sub_dir = os.path.join(output_dir, dir)
            #创建子目录路径
            os.makedirs(output_sub_dir, exist_ok=True)
            # 获取所有图片文件
            image_files = [f for f in os.listdir(os.path.join(root, dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for filename in tqdm(image_files, desc="Processing Images"):
                image_path = os.path.join(root, dir, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

                # 生成多次增强结果
                for i in range(augment_times):
                    # 应用变换并获取结果
                    augmented = transform(image=image)
                    # 提取增强后的图像
                    aug_image = augmented["image"]

                    # 保存增强后的图像
                    base_name = os.path.splitext(filename)[0]
                    output_path = os.path.join(output_sub_dir, f"{base_name}_aug{i}.jpg")
                    cv2.imwrite(output_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))




# 使用示例
batch_augment_albumentations(
    input_dir=r'e:\Code\Github\deeplearning\GoogLeNet-3\beads_origin',
    output_dir=r'e:\Code\Github\deeplearning\GoogLeNet-3\beads_aug',
    augment_times=200
)