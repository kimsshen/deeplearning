import os
from shutil import copy
import random


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
            copy(image_path, val_images_dir)
            # 将选中的图像标签复制到labels/val
            copy(label_path, val_labels_dir)

        # 其余的图像保存在训练集train中
        else:
            # 将选中的图像复制到images/val
            copy(image_path, train_images_dir)
            # 将选中的图像标签复制到labels/val
            copy(label_path, train_labels_dir)

    print("Totoal num of sample is: " + str(num))
    print("Totoal num of train sample is: " + str(num-len(eval_index)))
    print("Totoal num of val sample is: " + str(len(eval_index)))



if __name__ == '__main__':
    # 图像和标签文件夹
    image_path = "./data_aug/images"  #增强后的样本
    label_path = "./data_aug/labels"  #增强后的样本标签
    save_path = "./data"    # 结果保存位置路径，可以是一个不存在的文件夹

    split_rate=0.1

    print("Start the data partition!")
    # 运行
    runPartitionData(image_path, label_path, save_path,split_rate)
    print("Complete all the data partition!")
