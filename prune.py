#-*-coding:utf-8-*-
import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
from time import time

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


# fine-tune model
def fineTune(model):

    return state



# add 1x1 layers
def add1x1layers(convNumList):
    return AddLayerAlexNet(convNumList)

# reload paras
def reloadParam(model, state, convNumList):


# only train 1x1 layers and add L1  Regularization
def train1x1withL1(model, l1, conv1x1Lr, momentum)
    

    return [[],[]]

def prune_layers(model, threshold):

    return [[],[]]

 
def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def prune_vgg16_conv_layer(model, layer_index, filter_index):
    _, conv = model.features._modules.items()[layer_index]
    next_conv = None
    offset = 1

    while layer_index + offset <  len(model.features._modules.items()):
        res =  model.features._modules.items()[layer_index+offset]
        if isinstance(res[1], torch.nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset = offset + 1
    
    new_conv = \
        torch.nn.Conv2d(in_channels = conv.in_channels, \
            out_channels = conv.out_channels - 1,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups,
            bias = conv.bias)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    bias_numpy = conv.bias.data.cpu().numpy()

    bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index : ] = bias_numpy[filter_index + 1 :]
    new_conv.bias.data = torch.from_numpy(bias).cuda()

    if not next_conv is None:
        next_new_conv = \
            torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
                out_channels =  next_conv.out_channels, \
                kernel_size = next_conv.kernel_size, \
                stride = next_conv.stride,
                padding = next_conv.padding,
                dilation = next_conv.dilation,
                groups = next_conv.groups,
                bias = next_conv.bias)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index, layer_index+offset], \
                    [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features

    else:
        #Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index], \
                    [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index  + 1

        if old_linear_layer is None:
            raise BaseException("No linear laye found in classifier")
        params_per_input_channel = old_linear_layer.in_features / conv.out_channels

        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
                old_linear_layer.out_features)
        
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()        

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel :] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel :]
        
        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model

if __name__ == '__main__':
    model = models.vgg16(pretrained=True)
    model.train()

    t0 = time.time()
    model = prune_conv_layer(model, 28, 10)
print "The prunning took", time.time() - t0