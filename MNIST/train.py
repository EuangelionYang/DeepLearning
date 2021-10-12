# -*- coding: utf-8 -*-
# @Time : 2021/10/7 20:38
# @Author : YangYu
# @Email: yangyu.cs@outlook.com
# @File : train.py
# @Software: PyCharm
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from model import Model


def preprocess():
    """
    预处理
    1.下载MNIST数据集
    2.装载MNIST数据集
    :return: trainLoader,testLoader
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    # 训练数据集
    train_data = datasets.MNIST('../data', train=True, download=True,
                                transform=transform)
    # 测试数据集
    test_data = datasets.MNIST('../data', train=False, download=True,
                               transform=transform)
    # 每次训练64幅图片
    trainLoader = DataLoader(train_data, batch_size=64)
    # 每次测试1000幅图片
    testLoader = DataLoader(test_data, batch_size=1000)
    return trainLoader, testLoader


train_dataloader, test_dataloader = preprocess()
train_data_size = len(train_dataloader.dataset)
test_data_size = len(test_dataloader.dataset)
model = Model()

# loss函数
loss_func = nn.CrossEntropyLoss()
# 学习率
learning_rate = 1e-2
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0
# 训练轮次
epoch = 10
writer = SummaryWriter("../MNIST_logs")
for i in range(epoch):
    for data in train_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_func(outputs, targets)
        optimizer.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            writer.add_scalar("train_loss_ReLU", loss.item(), total_train_step)
            print("Epoch: {}, Step: {}, loss: {:.4f}".format(i + 1, total_train_step, loss.item()))

    total_test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # print(targets.shape)
            outputs = model(imgs)
            loss = loss_func(outputs, targets)
            total_test_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum()
    accuracy = correct / len(test_dataloader.dataset)
    print("Epoch: {}, Total Loss:{:.4f}, Accuracy: {:.2f}%".format(i + 1,
                                                                   total_test_loss,
                                                                   100 * accuracy))
    writer.add_scalar("test_total_loss_ReLU", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy_ReLU", accuracy, total_test_step)
    total_test_step += 1
    torch.save(model.state_dict(), "MNIST_ReLU{}.m".format(i + 1))
writer.close()
