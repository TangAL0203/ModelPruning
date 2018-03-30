#-*-coding:utf-8-*-
import torch
from torch.autograd import Variable
from torchvision import models
import sys
import os
import getpass
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.sgd import SGD
import utils.dataset as dataset
import argparse
from operator import itemgetter
from heapq import nsmallest
from time import time


import logging
#======================generate logging imformation===============
log_path = './log'
if not os.path.exists(log_path):
    os.mkdir(log_path)

# you should assign log_name first such as mobilenet_resnet50_CIFAR10.log
log_name = 'alexnetPrune.log'
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
        filename = './1x1models_fromScratch/' + '1x1fromScratch'+args.arch + '_' + args.data_name + '_' + str(acc) + '.pth'
        torch.save(model.state_dict(), filename)
    print("Finished training.")
    return model.state_dict()


# 训练一个epoch,测试一次
def train_val_test(model, train_loader, val_loader, test_loader, optimizer=None, epoches=15):
    print("Start training.")
    if optimizer is None:
        optimizer = optim.SGD(model.classifier.parameters(), lr = 0.04, momentum=0.9)

    for i in range(epoches):
        model.train()
        infoLogger.info("Epoch: "+str(i))
        train_epoch(model, train_loader, optimizer)
        acc = val_test(model, val_loader, test_loader)
        # filename = './1x1models/weightDecay5e-3/' + 'finetuned-weight_decay1-5e-3' + args.arch + '_' + args.data_name + '_' + str(acc) + '.pth'
        # torch.save(model.state_dict(), filename)
        state = model.state_dict()
        s1x1ParaName = ['features12.0.weight', 'features22.0.weight', 'features32.0.weight', 'features42.0.weight', 'features52.0.weight']
        for name in s1x1ParaName:
            sumStr = name+'  sum  is: '+str(torch.abs(state[name]).sum())
            meanStr = name+'  mean is: '+str(torch.mean(state[name]))
            stdStr = name+'  std  is: '+str(torch.std(state[name]))
            infoLogger.info(sumStr)
            infoLogger.info(meanStr)
            infoLogger.info(stdStr)
    infoLogger.info("Finished training.")
    return state


## suppose you load model with 1x1 layers and the model has beeb fine-tuned
def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Prunning Experiment')
    parser.add_argument('--arch', metavar='ARCH', default='alexnet', help='model architecture')
    parser.add_argument('--data_name', metavar='DATA_NAME', type=str, default='Flower102', help='dataset name')
    parser.add_argument('--breakpoint', default=False, action='store_true', help='choose if train from checkpoint or not')
    parser.add_argument('--l1', default=5e-3, type=float, help='set the l1 Regularization weight')
    parser.add_argument('--threshold', default=1e-1, type=float, help='set the threshold value')
    parser.add_argument('--conv1x1Lr', default=1e-1, type=float, help='set the learning rate of 1x1 layers')
    parser.add_argument('--convLr', default=0, type=float, help='set the learning rate of conv layers')
    parser.add_argument('--fcLr', default=1e-3, type=float, help='set the learning rate of fc layers')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--prune', default=False, action='store_true', help='choose prune or only fine-tune')
    args = parser.parse_args()
    return args

# model = OrigAddLayerAlexNet(102) 
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



# reload paras
# state: the previous OrderedDict of model
def reloadParam(num_classes, state, convIndexList=[[]]):
    global use_gpu
    if len(convIndexList[0])==0:
        convIndexList = []
        convIndexList.append(range(64))
        convIndexList.append(range(192))
        convIndexList.append(range(384))
        convIndexList.append(range(256))
        convIndexList.append(range(256))


    model = add1x1layers(num_classes, convIndexList)
    now_state = model.state_dict()
    npIndexList = [np.array(i) for i in convIndexList]
    # convNumList = [len(i) for i in convIndexList]
    convKeys = ['features11.0.weight',
                'features21.0.weight',
                'features31.0.weight',
                'features41.0.weight',
                'features51.0.weight']
    convBiasKeys = ['features11.0.bias',
                    'features21.0.bias',
                    'features31.0.bias',
                    'features41.0.bias',
                    'features51.0.bias']
    fcKeys = ['classifier.1.weight',
              'classifier.4.weight',
              'classifier.6.weight']
    fcBiasKeys = ['classifier.1.bias',
                  'classifier.4.bias',
                  'classifier.6.bias']
    conv1x1Keys = ['features12.0.weight',
               'features22.0.weight',
               'features32.0.weight',
               'features42.0.weight',
               'features52.0.weight']
    for i, key in enumerate(convKeys):
        if i==0:
            inIndex = np.array([0,1,2])
            outIndex = npIndexList[i]
            now_state[key].copy_(state[key][outIndex])
        else:
            inIndex = npIndexList[i-1]
            outIndex = npIndexList[i]
            now_state[key].copy_(state[key][outIndex][:,inIndex])

    for i, key in enumerate(convBiasKeys):
        outIndex = npIndexList[i]
        now_state[key].copy_(state[key][outIndex])
    # only first first layer need to change
    fcIndex = []
    for i in npIndexList[-1]:
        temp = range(i*36, (i+1)*36)
        fcIndex +=temp
    fcIndex = np.array(fcIndex)
    now_state[fcKeys[0]].copy_(state[fcKeys[0]][:,fcIndex])
    now_state[fcKeys[1]].copy_(state[fcKeys[1]])
    now_state[fcKeys[2]].copy_(state[fcKeys[2]])
    for i, key in enumerate(fcBiasKeys):
        now_state[key].copy_(state[key])

    # set 1x1 layers weights equal 1
    for i, key in enumerate(conv1x1Keys):
        shape = now_state[key].shape
        now_state[key].copy_(torch.ones(shape))

    if use_gpu:
        model = model.cuda()
        print("Use GPU!")
    else:
        print("Use CPU!")

    return model


# only train 1x1 layers and add L1  Regularization
def train1x1withL1(model, data_name='Flower102', threshold=0.1, l1=5e-3, conv1x1Lr=1e-1, momentum=0.9, epochs=15):
    global train_loader, val_loader, test_loader
    optimizer = SGD([
                {'params': model.features12.parameters()},
                {'params': model.features22.parameters()},
                {'params': model.features32.parameters()},
                {'params': model.features42.parameters()},
                {'params': model.features52.parameters()},
            ], weight_decay1=l1, lr=conv1x1Lr, momentum=momentum)

    if data_name=="Flower102":
        state = train_val_test(model, train_loader, val_loader, test_loader, optimizer=optimizer, epoches=epochs)
    else:
        state = train_test(model, train_loader, test_loader, optimizer=optimizer, epoches=epochs)

    s1x1ParaName = ['features12.0.weight', 'features22.0.weight', 'features32.0.weight', 'features42.0.weight', 'features52.0.weight']

    convIndexList = []

    for name in s1x1ParaName:
        para = state[name]
        para = torch.squeeze(torch.squeeze(torch.squeeze(para,1),1),1)
        temp = []
        for index,value in enumerate(para):
            if abs(value)<=threshold:
                temp.append(index)
        convIndexList.append(temp)


    return convIndexList

def prune_layers(model, threshold):


    return [[],[]]


# fine-tune model
def fineTune(model):
    global train_loader, val_loader, test_loader
    optimizer = optim.SGD(model.classifier.parameters(), lr = 0.001, momentum=0.9)
    train_val_test(model, train_loader, val_loader, test_loader, optimizer=optimizer, epoches=10)

    return state




if __name__ == '__main__':

    args = get_args()
    data_name = args.data_name
    breakpoint = args.breakpoint
    l1 = args.l1
    threshold = args.threshold
    conv1x1Lr = args.conv1x1Lr
    convLr = args.convLr
    fcLr = args.fcLr
    momentum = args.momentum
    prune = args.prune

    if 'Flower102' in args.data_name:
        train_path = "./Flower102/train"
        test_path = "./Flower102/test"
        val_path = "./Flower102/val"
    elif 'Birds200' in args.data_name:
        train_path = "./Birds200/train"
        test_path = "./Birds200/test"
    elif 'catdog' in args.data_name:
        train_path = "./CatDog/train"
        test_path = "./CatDog/test"
    # global train_path, test_path
    if 'Flower102' in train_path:
        train_loader = dataset.train_loader(train_path, batch_size=train_batch_size, num_workers=4, pin_memory=True)
        val_loader = dataset.test_loader(val_path, batch_size=1, num_workers=4, pin_memory=True)
        test_loader = dataset.test_loader(test_path, batch_size=1, num_workers=4, pin_memory=True)
    elif 'Birds200' in train_path:
        train_loader = dataset.train_loader(train_path, batch_size=train_batch_size, num_workers=4, pin_memory=True)
        test_loader = dataset.test_loader(test_path, batch_size=1, num_workers=4, pin_memory=True)
    elif 'catdog' in args.data_name:
        train_loader = dataset.train_loader(train_path, batch_size=train_batch_size, num_workers=4, pin_memory=True)
        test_loader = dataset.test_loader(test_path, batch_size=1, num_workers=4, pin_memory=True)

    infoLogger.info("dataset is: "+args.data_name)

    if not breakpoint:
        state = torch.load('./1x1models/origin1x1alexnet_Flower102_0.789.pth')
        if data_name=='Flower102':
            num_classes = 102
            model = reloadParam(num_classes, state)
        elif data_name=='Birds200':
            num_classes = 200
            model = reloadParam(num_classes, state)
    else:
        pass
    t0 = time()

    state = model.state_dict()
    convIndexList = train1x1withL1(model, data_name=data_name, threshold=threshold, l1=l1, conv1x1Lr=conv1x1Lr, momentum=momentum, epochs=15)
    model = reloadParam(num_classes, state, convIndexList=convIndexList)
    fineTune(model)

    print "The prunning took", time() - t0