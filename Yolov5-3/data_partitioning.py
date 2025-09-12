import os
import shutil
from shutil import copy
import random
import cv2

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def runPartitionData(images_dir,labels_dir,save_path,split_rate):
    train_images_dir = mkdir(save_path + '/images/train')
    val_images_dir = mkdir(save_path + '/images/val')
    train_labels_dir = mkdir(save_path + '/labels/train')
    val_labels_dir = mkdir(save_path + '/labels/val')

    images = os.listdir(images_dir)  # iamges 列表存储了该目录下所有图像的名称
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))  # 从images列表中随机抽取 k 个图像名称
    for index, image in enumerate(images):

        #图片的全路径
        image_path = os.path.join(images_dir, image)
        base_name, extension = os.path.splitext(image)
        #获取标签路径
        label_path = os.path.join(labels_dir , base_name + ".txt")

        # eval_index 中保存验证集val的图像名称
        if image in eval_index:
            # 将选中的图像复制到images/val
            transferImg(image_path,val_images_dir+'/'+image)
            # 将选中的图像标签复制到labels/val
            copy(label_path, val_labels_dir)

        # 其余的图像保存在训练集train中
        else:
            # 将选中的图像复制到images/val
            transferImg(image_path, train_images_dir + '/' + image)
            # 将选中的图像标签复制到labels/val
            copy(label_path, train_labels_dir)

    print("Totoal num of sample is: " + str(num))
    print("Totoal num of train sample is: " + str(num-len(eval_index)))
    print("Totoal num of val sample is: " + str(len(eval_index)))


def transferImg(gray_img_path,rgb_img_path):
    # 1. 读取灰度图（强制按灰度读，避免意外）
    img_gray = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)

    # 2. 转为三通道 RGB（实际上是 R=G=B=灰度值）
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # 3. 保存为新图片（默认保存为 BGR，但视觉效果一样）
    cv2.imwrite(rgb_img_path, img_rgb)



if __name__ == '__main__':
    # 图像和标签文件夹
    image_path = "./data_aug/images"  #增强后的样本
    label_path = "./data_aug/labels"  #增强后的样本标签
    save_path = "./data"    # 结果保存位置路径，可以是一个不存在的文件夹

    #清理目录
    try:
        # 递归删除目录及其所有内容
        shutil.rmtree(save_path)
        print(f"成功删除目录: {save_path}")
    except Exception as e:
        print(f"未知错误: {e}")

    #验证集的比例为1/10
    split_rate=0.1

    print("Start the data partition!")
    # 运行
    runPartitionData(image_path, label_path, save_path,split_rate)
    print("Complete all the data partition!")
