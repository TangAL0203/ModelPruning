#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import utils.dataset as dataset
import torchvision
import math
import os
import getpass
import shutil
import argparse

parser = argparse.ArgumentParser(description='Pytorch Distillation Experiment')
parser.add_argument('--arch', metavar='ARCH', default='alexnet', type=str, help='model architecture')
parser.add_argument('--data_name', metavar='DATA_NAME', type=str, default='Flower102', help='dataset name')

args = parser.parse_args()


import logging
#======================generate logging imformation===============
log_path = './log'
if not os.path.exists(log_path):
    os.mkdir(log_path)

# you should assign log_name first such as mobilenet_resnet50_CIFAR10.log
log_name = 'Alexnet_Flowers102.log'
TrainInfoPath = os.path.join(log_path, log_name)
# formater
formatter = logging.Formatter('%(levelname)s %(message)s')
# cmd Handler 
cmdHandler = logging.StreamHandler() 
# File Handler including info
infoFileHandler = logging.FileHandler(TrainInfoPath, mode='w')
infoFileHandler.setFormatter(formatter)
# info Logger
infoLogger = logging.getLogger('info') 
infoLogger.setLevel(logging.DEBUG) 
infoLogger.addHandler(cmdHandler)
infoLogger.addHandler(infoFileHandler)

if getpass.getuser() == 'tsq':
    train_batch_size = 8
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_batch_size = 32

use_gpu = torch.cuda.is_available()
num_batches = 0

###### you should set data path here ###### 
## train and val on the Flower102 dataset
# train_path = "./Flower102/train"
# test_path = "./Flower102/test"
# val_path = "./Flower102/val"
## train and val on the Birds200 dataset
# train_path = "./Birds200/train"
# test_path = "./Birds200/test"

def val_test(model, val_loader, test_loader):
    model.eval()
    val_correct = 0
    val_total = 0
    test_correct = 0
    test_total = 0

    for i, (batch, label) in enumerate(val_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        val_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        val_total += label.size(0)

    for i, (batch, label) in enumerate(test_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        test_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        test_total += label.size(0)

    infoLogger.info("Val  Accuracy :"+str(round( float(val_correct) / val_total , 3 )))
    infoLogger.info("Test Accuracy :"+str(round( float(test_correct) / test_total , 3 )))
    model.train()
    return round( float(test_correct) / test_total , 3 )

def test(model, test_loader):
    model.eval()
    test_correct = 0
    test_total = 0

    for i, (batch, label) in enumerate(test_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] # 返回模型预测概率最大的标签
        test_correct += pred_label.cpu().eq(label).sum() # label为torch.LongTensor类型
        test_total += label.size(0)

    infoLogger.info("Test Accuracy :"+str(round( float(test_correct) / test_total , 3 )))
    model.train()
    return round( float(test_correct) / test_total , 3 )

def train_batch(model, optimizer, batch, label): 
    optimizer.zero_grad() # 
    input = Variable(batch)
    output = model(input)
    criterion = torch.nn.CrossEntropyLoss()
    criterion(output, Variable(label)).backward() 
    optimizer.step()
    return criterion(output, Variable(label)).data

def train_epoch(model, train_loader, optimizer=None):
    global num_batches
    for batch, label in train_loader:
        loss = train_batch(model, optimizer, batch.cuda(), label.cuda())
        if num_batches%1 == 0:
            infoLogger.info('%23s%-9s%-13s'%('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
        num_batches +=1

# 训练一个epoch,测试一次
def train_test(model, train_loader, test_loader, optimizer=None, epoches=10):
    print("Start training.")
    if optimizer is None:
        optimizer = optim.SGD(model.classifier.parameters(), lr = 0.001, momentum=0.9)

    for i in range(epoches):
        model.train()
        print("Epoch: ", i)
        train_epoch(model, train_loader, optimizer)
        acc = test(model, test_loader)
        filename = './models/' + args.arch + '_' + args.data_name + '_' + str(acc) + '.pth'
        torch.save(model.state_dict(), filename)
    print("Finished training.")

# 训练一个epoch,测试一次
def train_val_test(model, train_loader, val_loader, test_loader, optimizer=None, epoches=10):
    print("Start training.")
    if optimizer is None:
        optimizer = optim.SGD(model.classifier.parameters(), lr = 0.0004, momentum=0.9)

    for i in range(epoches):
        model.train()
        infoLogger.info("Epoch: "+str(i))
        train_epoch(model, train_loader, optimizer)
        acc = val_test(model, val_loader, test_loader)
        filename = './models/' + args.arch + '_' + args.data_name + '_' + str(acc) + '.pth'
        torch.save(model.state_dict(), filename)
    infoLogger.info("Finished training.")


# 初始化模型参数
#　从0开始训练一个二分类器
# 对conv层和全连接层参数初始化
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0, std=1)
        m.bias.data.zero_()

class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedAlexNet, self).__init__()
        model = torchvision.models.alexnet(pretrained=True)
        self.features = model.features
        for param in self.features.parameters():
            param.requires_grad = False

        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    # you should set data path on the top
    global train_path, val_path, test_path
    global train_batch_size
    if 'Flower102' in args.data_name:
        train_path = "./Flower102/train"
        test_path = "./Flower102/test"
        val_path = "./Flower102/val"
    elif 'Birds200' in args.data_name:
        train_path = "./Birds200/train"
        test_path = "./Birds200/test"

    # global train_path, test_path
    if 'Flower102' in train_path:
        model = ModifiedAlexNet(102) 
        train_loader = dataset.train_loader(train_path, batch_size=train_batch_size, num_workers=4, pin_memory=True)
        val_loader = dataset.test_loader(val_path, batch_size=1, num_workers=4, pin_memory=True)
        test_loader = dataset.test_loader(test_path, batch_size=1, num_workers=4, pin_memory=True)
    elif 'Birds200' in train_path:
        model = ModifiedAlexNet(200)
        train_loader = dataset.train_loader(train_path, batch_size=train_batch_size, num_workers=4, pin_memory=True)
        test_loader = dataset.test_loader(test_path, batch_size=1, num_workers=4, pin_memory=True)
    if use_gpu:
        model = model.cuda()
        print("Use GPU!")
    else:
        print("Use CPU!")

    checkpoint = torch.load('./models/alexnet_Flower102_0.787.pth')
    model.load_state_dict(checkpoint, strict=True)

    if 'Flower102' in train_path:
        train_val_test(model, train_loader, val_loader, test_loader, optimizer=None, epoches=50)
    elif 'Birds200' in train_path:
        train_test(model, train_loader, test_loader, optimizer=None, epoches=10)

if __name__ == "__main__":
    main()


## usage
# python alexnet.py --arch alexnet --data_name Flower102
# python alexnet.py --arch alexnet --data_name Birds200