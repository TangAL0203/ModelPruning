#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import utils.dataset as dataset
import torchvision
import torchvision.datasets.folder as folder
import math
import os
import getpass
import shutil
import argparse


parser = argparse.ArgumentParser(description='Pytorch Distillation Experiment')
parser.add_argument('--arch', metavar='ARCH', default='alexnet', help='model architecture')
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

class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedAlexNet, self).__init__()
        model = torchvision.models.alexnet(pretrained=False)
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

# add a 1x1 depthwise conv layer between former conv layers
class AddLayerAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AddLayerAlexNet, self).__init__()
        model = torchvision.models.alexnet(pretrained=False)
        layer_list1 = []
        layer_list2 = []
        layer_list3 = []
        layer_list4 = []
        layer_list5 = []
        for i, layer in enumerate(model.features):
            if i<=2:
                layer_list1.append(layer)
                if i==0:
                    channels = layer.out_channels
                if i==2:
                    layer_list1.append(nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,groups=channels,bias=False))
            elif i<=5:
                layer_list2.append(layer)
                if i==3:
                    channels = layer.out_channels
                if i==5:
                    layer_list2.append(nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,groups=channels,bias=False))
            elif i<=7:
                layer_list3.append(layer)
                if i==6:
                    channels = layer.out_channels
                if i==7:
                    layer_list3.append(nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,groups=channels,bias=False))
            elif i<=9:
                layer_list4.append(layer)
                if i==8:
                    channels = layer.out_channels
                if i==9:
                    layer_list4.append(nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,groups=channels,bias=False))
            elif i<=12:
                layer_list5.append(layer)
                if i==10:
                    channels = layer.out_channels
                if i==12:
                    layer_list5.append(nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1,groups=channels,bias=False))

        # add a 1x1 depthwise conv layer and a mask layer
        self.features1 = nn.Sequential(*layer_list1)
        self.features2 = nn.Sequential(*layer_list2)
        self.features3 = nn.Sequential(*layer_list3)
        self.features4 = nn.Sequential(*layer_list4)
        self.features5 = nn.Sequential(*layer_list5)

        for param in self.features1.parameters():
            param.requires_grad = False
        list(self.features1.parameters())[-1].requires_grad = True

        for param in self.features2.parameters():
            param.requires_grad = False
        list(self.features2.parameters())[-1].requires_grad = True

        for param in self.features3.parameters():
            param.requires_grad = False
        list(self.features3.parameters())[-1].requires_grad = True

        for param in self.features4.parameters():
            param.requires_grad = False
        list(self.features4.parameters())[-1].requires_grad = True

        for param in self.features5.parameters():
            param.requires_grad = False
        list(self.features5.parameters())[-1].requires_grad = True

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
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# for alxenet, 1x1 layers indexs should be [2 5 8 11 14]
# all layers indexs is [0 1 2 .... 20] 21=16+5
def my_load_state_dict(model, state_dict):
    own_state = model.state_dict()
    j = 0
    for i in range(21):
        if i in [2, 5, 8, 11, 14]:
            shape = own_state.items()[i][1].shape
            own_state.items()[i][1].copy_(torch.ones(shape)) # init with 1
        else:
            own_state.items()[i][1].copy_(state_dict.items()[j][1])
            j +=1


def main():
    checkpoint = torch.load('./models/alexnet_Flower102_0.789.pth')
    # predict 
    model = AddLayerAlexNet(102)
    my_load_state_dict(model, checkpoint)
    classes, class_to_idx = folder.find_classes('./Flower102/train/')
    # load one jpg, convert to Variable
    root = '/home/smiles/ModelPruning/Flower102/test'
    path = '/home/smiles/ModelPruning/Flower102/test/44/image_07150.jpg'
    label_id = class_to_idx[path.split('/')[-2]]
    img, null = dataset.get_single_img(root, path, train=False)
    inputVariable = Variable(img)
    model.eval()
    output = model(inputVariable)
    pred_label = output.data.max(1)[1]
    print "true label is: ",label_id
    print "pred label is: ",pred_label.numpy()[0]



if __name__ == "__main__":
    main()

