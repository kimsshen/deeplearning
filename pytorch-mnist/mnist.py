import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 检查CUDA设备是否可用，然后选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(torch.nn.Module):#继承
    def __init__(self):#初始化方法
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)  #这是第一个全连接层（Fully Connected Layer），它将输入的特征维度从 28 * 28（通常是28x28像素的图像展平后的长度）转换为 64 个输出单元。
        self.fc2 = torch.nn.Linear(64, 64)#这是第二个全连接层，将前一个层的 64 个特征映射到另外 64 个特征。
        self.fc3 = torch.nn.Linear(64, 64)#这是第三个全连接层，将 64 个特征映射到另外 64 个特征。
        self.fc4 = torch.nn.Linear(64, 10)#这是第四个全连接层，将 64 个特征映射到 10 个输出特征，这通常是一个分类任务的输出层，10 代表可能的分类类别数量（例如，对于手写数字分类，可能对应 0-9 的数字）。

    def forward(self, x):
        x=torch.nn.functional.relu(self.fc1(x))#self.fc1(x)：将输入数据 x 传递到第一个全连接层 fc1。fc1 的输出维度是 [N, 64]。
        #torch.nn.functional.relu()：对 fc1 的输出应用 ReLU 激活函数。ReLU（Rectified Linear Unit）将负值转换为 0，正值保持不变。这样可以引入非线性特性，并帮助网络更好地学习复杂模式。
        x=torch.nn.functional.relu(self.fc2(x))
        x=torch.nn.functional.relu(self.fc3(x))
        x=torch.nn.functional.log_softmax(self.fc4(x), dim=1)#log_softmax 是 softmax 的对数版本。这个函数对每个样本的得分进行归一化处理，并计算其对数概率。dim=1 指定在特征维度（即类别维度）上进行操作。
        return x

def get_data_loader(is_train):
    to_tensor=transforms.Compose([transforms.ToTensor()])
    data_set=MNIST(root='mnist_data/', train=is_train, download=True, transform=to_tensor)
    return DataLoader(data_set, batch_size=15, shuffle=True, pin_memory=True)

def evaluate(test_data,net):
    net.eval()  # 将模型设置为评估模式
    n_correct=0
    n_total=0
    #n_correct：用于记录模型预测正确的样本数量。
    #n_total：用于记录测试样本的总数量。
    with torch.no_grad():#禁用梯度计算。在评估模型时，我们不需要计算梯度，因为不会进行反向传播。使用 torch.no_grad() 可以节省内存并提高计算效率。
        for (x,y) in test_data:
            x, y = x.to(device), y.to(device)  # 移动数据到设备
            outputs = net.forward(x.view(-1,28*28))
            for i,output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total+=1
    return n_correct / n_total

def main():
    train_data = get_data_loader(is_train=True) #获取训练数据
    test_data = get_data_loader(is_train=False) #获取测试数据
    net = Net().to(device)# 将模型移动到选择的设备


    print("initial accuracy:",evaluate(test_data,net))
    optimizer =torch.optim.Adam(net.parameters(),lr=0.001)

    for epoch in range(2):
        net.train()  # 将模型设置为训练模式
        for(x,y) in train_data:
            x, y = x.to(device), y.to(device)  # 移动数据到设备
            net.zero_grad()#初始化
            outputs = net.forward(x.view(-1,28*28))#正向传播
            loss=torch.nn.functional.nll_loss(outputs,y)#计算差值
            loss.backward()#反向误差传播
            optimizer.step()#优化网络参数
        print("epoch:",epoch,"accuracy:",evaluate(test_data,net))

    plt.figure(figsize=(10, 5))
    for n, (x, _) in enumerate(test_data):
        if n > 3:
            break
        x = x.to(device)
        prediction = torch.argmax(net(x.view(-1, 28 * 28)))
        plt.subplot(2, 2, n + 1)
        plt.imshow(x[0].view(28, 28).cpu().numpy(), cmap='gray')  # 确保在绘图前数据在CPU上
        plt.title("Prediction: " + str(prediction.item()))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()

