from torchvision import datasets


class MyDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        # 获取原始样本元组
        sample = super(MyDataset, self).__getitem__(index)
        img, label = sample

        # 获取路径信息
        img_path = self.samples[index][0]

        return img, label, img_path