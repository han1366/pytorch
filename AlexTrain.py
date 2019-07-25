'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

import math

from Alex.AlexNet import *
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# 获取数据集，并先进行预处理
print('==> Preparing data..')
# 图像预处理和增强
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),                                                        #转化成张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  #数据归一化处理，需要先将数据转化成张量

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

#参数值初始化
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weigth.data.fill_(1)
        m.bias.data.zero_()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

n_output = 10
net = AlexNet(10)

# 如果GPU可用，使用GPU
if use_cuda:
    # move param and buffer to GPU
    net.cuda()
    # parallel use GPU
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()-1))
    # speed up slightly
    cudnn.benchmark = True


# 定义度量和优化
criterion = nn.CrossEntropyLoss()                      #交叉熵验证
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)      #随机梯度下降

# 训练阶段
def train(epoch):
    print('\nEpoch: %d' % epoch)
    # switch to train mode
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # batch 数据
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 将数据移到GPU上
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # 先将optimizer梯度先置为0
        optimizer.zero_grad()
        # Variable表示该变量属于计算图的一部分，此处是图计算的开始处。图的leaf variable

        inputs, targets = Variable(inputs), Variable(targets)
        # 模型输出
        outputs = net(inputs)
        # 计算loss，图的终点处
        loss = criterion(outputs, targets)
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 注意如果你想统计loss，切勿直接使用loss相加，而是使用loss.item()。因为loss是计算图的一部分，如果你直接加loss，代表total loss同样属于模型一部分，那么图就越来越大
        train_loss += loss.item()
        # 数据统计
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 100 == 0:
            print('Train　Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx, batch_idx * len(inputs), len(trainloader.dataset),
                   100. * batch_idx / (len(trainloader)), loss.item()))

# 测试阶段
def test():
    # 先切到测试模型
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for inputs, targets in testloader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # loss is variable , if add it(+=loss) directly, there will be a bigger ang bigger graph.
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    test_loss /= len(testloader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))

net.apply(weight_init)

for epoch in range(1,11):
    print('----------------start train-----------------')
    train(epoch)
    print('----------------end train-----------------')

    print('----------------start test-----------------')
    test()
    print('----------------end test-----------------')




torch.save(net.state_dict(), 'AlexNetparams.pkl')
