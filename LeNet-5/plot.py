from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data

train_data=FashionMNIST(root='./data',
                        train=True,
                        transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor]),
                        download=True)
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

