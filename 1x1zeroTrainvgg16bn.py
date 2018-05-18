#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import utils.dataset as dataset
from utils.sgd import SGD
import torchvision
import torchvision.datasets.folder as folder
import math
import os
import getpass
import shutil
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch finetune or zero train 1x1 model')
    parser.add_argument('--arch', metavar='ARCH', default='vgg16bn', help='model architecture')
    parser.add_argument('--data_name', metavar='DATA_NAME', type=str, default='Flower102', help='dataset name')
    parser.add_argument('--zero_train', default=False, action='store_true', help='choose if train from Scratch or not')
    parser.add_argument('--checkpoint', default=False, action='store_true', help='choose if train from checkpoint')
    args = parser.parse_args()
    return args

args = get_args()
arch = args.arch
data_name = args.data_name
zero_train = args.zero_train
checkpoint = args.checkpoint


import logging
#======================generate logging imformation===============
log_path = './log'
if not os.path.exists(log_path):
    os.mkdir(log_path)

# you should assign log_name first such as mobilenet_resnet50_CIFAR10.log

log_name = '1x1'+arch+'Finetune'+args.data_name+'.log'
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
    train_batch_size = 64

use_gpu = torch.cuda.is_available()
num_batches = 0


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
        filename = './1x1models/finetune/' + args.arch + '1x1' + '_' + args.data_name + '_' + str(acc) + '.pth'
        s1x1ParaName = ['features12.0.weight', 'features22.0.weight', 'features32.0.weight', 'features42.0.weight', 'features52.0.weight']
        state = model.state_dict()
        for name in s1x1ParaName:
            sumStr = name+'  sum  is: '+str(torch.abs(state[name]).sum())
            meanStr = name+'  mean is: '+str(torch.mean(state[name]))
            stdStr = name+'  std  is: '+str(torch.std(state[name]))
            infoLogger.info(sumStr)
            infoLogger.info(meanStr)
            infoLogger.info(stdStr)

        torch.save(state, filename)
    print("Finished training.")

class AddLayerAlexNet(nn.Module):
    def __init__(self, num_classes=2, convNumList=[64, 192, 384, 256, 256]):
        super(AddLayerAlexNet, self).__init__()
        self.features11 = nn.Sequential(
                nn.Conv2d(3, convNumList[0], 11, 4, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, dilation=1),
            )
        self.features12 = nn.Sequential(
                nn.Conv2d(in_channels=convNumList[0],out_channels=convNumList[0],kernel_size=1,stride=1,groups=convNumList[0],bias=False),
            )
        self.features21 = nn.Sequential(
                nn.Conv2d(convNumList[0], convNumList[1], 5, 1, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, dilation=1),
            )
        self.features22 = nn.Sequential(
                nn.Conv2d(in_channels=convNumList[1],out_channels=convNumList[1],kernel_size=1,stride=1,groups=convNumList[1],bias=False),
            )
        self.features31 = nn.Sequential(
                nn.Conv2d(convNumList[1], convNumList[2], 3, 1, 1),
                nn.ReLU(inplace=True),
            )
        self.features32 = nn.Sequential(
                nn.Conv2d(in_channels=convNumList[2],out_channels=convNumList[2],kernel_size=1,stride=1,groups=convNumList[2],bias=False),
            )
        self.features41 = nn.Sequential(
                nn.Conv2d(convNumList[2], convNumList[3], 3, 1, 1),
                nn.ReLU(inplace=True),
            )
        self.features42 = nn.Sequential(
                nn.Conv2d(in_channels=convNumList[3],out_channels=convNumList[3],kernel_size=1,stride=1,groups=convNumList[3],bias=False),
            )
        self.features51 = nn.Sequential(
                nn.Conv2d(convNumList[3], convNumList[4], 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, dilation=1),
            )
        self.features52 = nn.Sequential(
                nn.Conv2d(in_channels=convNumList[4],out_channels=convNumList[4],kernel_size=1,stride=1,groups=convNumList[4],bias=False),
            )

        for param in self.features11.parameters():
            param.requires_grad = False
        for param in self.features12.parameters():
            param.requires_grad = True

        for param in self.features21.parameters():
            param.requires_grad = False
        for param in self.features22.parameters():
            param.requires_grad = True

        for param in self.features31.parameters():
            param.requires_grad = False
        for param in self.features32.parameters():
            param.requires_grad = True

        for param in self.features41.parameters():
            param.requires_grad = False
        for param in self.features42.parameters():
            param.requires_grad = True

        for param in self.features51.parameters():
            param.requires_grad = False
        for param in self.features52.parameters():
            param.requires_grad = True

        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(convNumList[4] * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features11(x)
        x = self.features12(x)
        x = self.features21(x)
        x = self.features22(x)
        x = self.features31(x)
        x = self.features32(x)
        x = self.features41(x)
        x = self.features42(x)
        x = self.features51(x)
        x = self.features52(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# add 1x1 layers
def add1x1layers(num_classes, convIndexList=[[]]):
    if len(convIndexList[0])==0:
        return AddLayerAlexNet(num_classes)
    convNumList = [len(i) for i in convIndexList]
    return AddLayerAlexNet(num_classes, convNumList)


# first fine-tune alexnet modefl from pre-trainde model on imagenet dataset
def firstreloadParam(num_classes):
    global use_gpu
    state = torchvision.models.alexnet(pretrained=True).state_dict()
    model = add1x1layers(num_classes)
    now_state = model.state_dict()

    oldconvKeys = ['features.0.weight',
                   'features.3.weight',
                   'features.6.weight',
                   'features.8.weight',
                   'features.10.weight']

    oldconvBiasKeys = ['features.0.bias',
                       'features.3.bias',
                       'features.6.bias',
                       'features.8.bias',
                       'features.10.bias']

    newconvKeys = ['features11.0.weight',
                   'features21.0.weight',
                   'features31.0.weight',
                   'features41.0.weight',
                   'features51.0.weight']

    newconvBiasKeys = ['features11.0.bias',
                       'features21.0.bias',
                       'features31.0.bias',
                       'features41.0.bias',
                       'features51.0.bias']

    conv1x1Keys = ['features12.0.weight',
                   'features22.0.weight',
                   'features32.0.weight',
                   'features42.0.weight',
                   'features52.0.weight']

    for i, key in enumerate(oldconvKeys):
        now_state[newconvKeys[i]].copy_(state[key])

    for i, key in enumerate(oldconvBiasKeys):
        now_state[newconvBiasKeys[i]].copy_(state[key])

    for key in conv1x1Keys:
        shape = now_state[key].shape
        now_state[key].copy_(torch.ones(shape))

    if use_gpu:
        model = model.cuda()
        print("Use GPU!")
    else:
        print("Use CPU!")

    return model

def finetune():
    global infoLogger
    global use_gpu, num_batches, train_batch_size

    if not zero_train:
        global train_path, test_path
        global train_batch_size
        if 'Flower102' in data_name:
            train_path = "./Flower102/train"
            test_path = "./Flower102/test"
            num_classes = 102
        elif 'Birds200' in data_name:
            train_path = "./Birds200/train"
            test_path = "./Birds200/test"
            num_classes = 200
        elif 'catdog' in data_name:
            train_path = "./CatDog/train"
            test_path = "./CatDog/test"
            num_classes = 2

        if not checkpoint:
            if use_gpu:
                print("Use GPU!")
                model = firstreloadParam(num_classes)
        else:
            checkpointPath = './1x1models/finetune/alexnet1x1_Flower102_0.821.pth'
            state = torch.load(checkpointPath)
            if use_gpu:
                print("Use GPU!")
                model = add1x1layers(num_classes).cuda()
            else:
                print("Use CPU!")
                model = add1x1layers(num_classes)
            model.load_state_dict(state) 
        train_loader = dataset.train_loader(train_path, batch_size=train_batch_size, num_workers=4, pin_memory=True)
        test_loader = dataset.test_loader(test_path, batch_size=1, num_workers=4, pin_memory=True)
        infoLogger.info("dataset is: "+args.data_name)
        optimizer = SGD([
                {'params': model.classifier.parameters()}   # 浅层1x1不好剪枝, 先冻住
            ], weight_decay2=5e-4, lr=1e-3, momentum=0.9)
        infoLogger.info("weight_decay2 is: "+str(5e-4))
        infoLogger.info("learning rate is: "+str(1e-3))
        train_test(model, train_loader, test_loader, optimizer=optimizer, epoches=20)


if __name__ == "__main__":
    finetune()



# usage 
# python 1x1zeroTrainalexnet.py
# python 1x1zeroTrainalexnet.py --checkpoint
