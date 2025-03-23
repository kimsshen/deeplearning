import copy
import time

import pandas as pd
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from  model import LeNet
import torch.nn as nn

def train_val_data_process():
    train_origin_data=FashionMNIST(root='./data',
                        train=True,
                        transform=transforms.Compose([transforms.Resize(size=32),transforms.ToTensor()]),
                        download=True)

    train_data, val_data = Data.random_split(train_origin_data,[round(0.8*len(train_origin_data)),round(0.2*len(train_origin_data))])


    train_dataloader = Data.DataLoader(dataset=train_data,
                               batch_size=128,
                               shuffle=True,
                               num_workers=8)
    val_dataloader = Data.DataLoader(dataset=val_data,
                               batch_size=128,
                               shuffle=True,
                               num_workers=8)
    return train_dataloader,val_dataloader

#训练模型
def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss()
    #将模型放入训练设备
    model = model.to(device)
    #赋值当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    #初始化参数
    best_acc = 0.0
    #训练集损失函数列表
    train_loss_all = []
    #训练集准确度列表
    train_acc_all = []
    #验证集损失函数列表
    val_loss_all = []
    #验证集准确度列表
    val_acc_all = []
    #当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("epoch {}/{}".format(epoch,num_epochs-1))
        print("-"*10)

        #初始化参数
        train_loss =0.0
        #训练数据正确样本数
        train_corrects = 0
        #验证集
        val_loss = 0.0
        val_corrects =0
        #训练集的样本数量
        train_num = 0
        #验证集的样本数量
        val_num = 0

        #对每一个mini-batch训练和计算
        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x= b_x.to(device)
            b_y= b_y.to(device)

            #设置模型为训练模式
            model.train()
            #前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)

            #查找每一行最大值对应的行标
            pre_lab = torch.argmax(output,dim=1)
            #计算每个批次的损失
            loss = criterion(output,b_y)

            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #利用梯度下降法进行参数更新
            optimizer.step()
            #loss.item()每个样本损失函数的平均值 *b_x.size(0)样本数量
            train_loss += loss.item() * b_x.size(0)

            train_corrects += torch.sum(pre_lab == b_y.data)
            #该轮次训练样本数量
            train_num +=b_x.size(0)
        #验证数据
        for step,(b_x,b_y) in enumerate(val_dataloader):
            #将特征放入到验证设备中
            b_x = b_x.to(device)
            #将标签放入到验证设备
            b_y = b_y.to(device)
            #设置模型为评估模式
            model.eval()
            #前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            #查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output,dim=1)
            #计算每一个batch的损失函数
            loss = criterion(output,b_y)

            #对损失函数进行累加
            val_loss += loss.item()*b_x.size(0)
            #如果预测准确，则朱雀度train_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            #当前用于验证的样本数量
            val_num += b_y.size(0)

        #计算每一轮的loss值和准确率
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)

        #验证集的loss值和准确率
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)

        print('{} Train Loss:{:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print('{} Val Loss:{:.4f} Val Acc: {:.4f}'.format(epoch,val_loss_all[-1],val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            #保存当前最高准确度
            best_acc = val_acc_all[-1];
            #保存当前的权重参数
            best_model_wts = copy.deepcopy(model.state_dict())
        #训练耗时
        time_use = time.time()-since
        print("训练和验证耗费时间{:.0f}m{:.0f}s".format(time_use//60,time_use%60))

    #选择最有参数
    #加载最高准确率下的模型参数
    torch.save(best_model_wts,'./best_model.pth')

    print(
        len(range(num_epochs)),
        len(train_loss_all),
        len(val_loss_all),
        len(train_acc_all),
        len(val_acc_all)
    )

    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all})
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"],train_process.train_acc_all,'ro-',label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__== "__main__":
    #将模型实例化
    LeNet = LeNet()
    train_dataloader,val_dataloader = train_val_data_process()
    train_process = train_model_process(LeNet,train_dataloader,val_dataloader,20)
    matplot_acc_loss(train_process)







