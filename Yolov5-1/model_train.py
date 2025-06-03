import os
import time
from pathlib import Path

import logging
import shutil

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("packaging_inspection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PackagingInspector")



def train_yolov5_model(epochs=100, batch_size=16, model_size='s'):
    """
    训练YOLOv5模型
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param model_size: 模型大小 (s, m, l, x)
    """
    logger.info("开始训练YOLOv5模型...")
    
    # 检查YOLOv5仓库是否存在
    # if not Path('yolov5').exists():
    #     logger.info("正在克隆YOLOv5仓库...")
    #     os.system('git clone https://github.com/ultralytics/yolov5')
    #     os.chdir('yolov5')
    #     os.system('conda install -r requirements.txt')
    #     os.chdir('..')
    
#     # 创建数据集配置文件
    dataset_yaml = 'dataset.yaml'

    str_time=time.strftime('%Y%m%d_%H%M%S')
    # 构建训练命令
    train_cmd = (
        f"python yolov5/train.py "
        f"--img 640 "
        f"--batch {batch_size} "
        f"--epochs {epochs} "
        f"--data {dataset_yaml} "
        f"--weights yolov5{model_size}.pt "
        f"--project packaging_models "
        f"--name yolov5{model_size}_{str_time} "
        f"--exist-ok "
        f"--cache ram "
        f"--optimizer AdamW "
        f"--cos-lr "
    )
    logger.info(f"执行训练命令: {train_cmd}")
    os.system(train_cmd)
    
    # 查找最佳模型
    model_dir = Path('packaging_models') / f"yolov5{model_size}_{str_time}"
    logger.info(f"model_dir:: {model_dir}")
    best_model = next(model_dir.glob('**/best*.pt'))
    model_path = Path('packaging_models') /f"best.pt"
    # 拷贝模型到外层目录
    shutil.copy2(best_model, model_path)

    return best_model,model_path

if __name__ == "__main__":


    # 训练模型 (如果需要)

    dataset_path = "data"
    best_model,model_path = train_yolov5_model(
        epochs=100,
        batch_size=32,
        model_size='s'
    )
    logger.info(f"训练完成! 最佳模型保存在: {best_model}"+ f",模型拷贝至: {model_path}")

    
