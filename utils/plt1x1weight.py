# -*- coding:utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

path1 = './finetuned-weight_decay1-5e-3alexnet_Flower102_0.769.pth'
path2 = './finetuned-weight_decay1-5e-3alexnet_Flower102_0.745.pth'
path3 = './finetuned-weight_decay1-5e-3alexnet_Flower102_0.721.pth'
path4 = './finetuned-weight_decay1-5e-3alexnet_Flower102_0.673.pth'
path5 = './finetuned-weight_decay1-5e-3alexnet_Flower102_0.655.pth'

pathList = [path1, path2, path3, path4, path5]
# names = ['features12.0.weight', 'features22.0.weight', 'features32.0.weight', 'features42.0.weight', 'features52.0.weight']
names = ['features22.0.weight', 'features32.0.weight', 'features42.0.weight', 'features52.0.weight']

for name in names:
    if not os.path.exists(name):
        os.mkdir(name)


for i,path in enumerate(pathList):
    para = torch.load(path)
    for name in names:
        root = './'+name+'/'
        features = para[name].cpu().numpy()
        features = features.reshape(features.shape[0],)
        if i==3:
            numTiny = np.where(features<=0.1)[0].shape[0]
            print 'tinyNum in '+name+' is: '+str(numTiny)
        plt.plot(features)
        saveName = root+'orig-'+str(i)+'-'+name+'.jpg'
        plt.savefig(saveName)
        #plt.show()
        features = (features-min(features))/(max(features)-min(features))
        plt.plot(features)
        saveName = root+'minmax-'+str(i)+'-'+name+'.jpg'
        plt.savefig(saveName)
        #plt.show()
        


