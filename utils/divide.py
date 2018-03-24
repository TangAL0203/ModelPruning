import numpy as np 
import scipy.io as sio
import shutil
import os
import os.path as osp

data = sio.loadmat('./setid.mat')
label = sio.loadmat('./imagelabels.mat')

pic_list = os.listdir('./jpg/')

kk = 0
for i in data['valid'][0]:
    if i<=9:
        temp = '0000'+str(i)
    elif i <=99:
        temp = '000'+str(i)
    elif i <=999:
        temp = '00'+str(i)
    elif i <=9999:
        temp = '0'+str(i)
    for j in pic_list:
        if str(temp) in j:
            kk = kk+1
            print kk
            shutil.copy(osp.join('./jpg/',j),osp.join('./val/',str(label['labels'][0][i-1]-1)))
            break
kk = 0
for i in data['trnid'][0]:
    if i<=9:
        temp = '0000'+str(i)
    elif i <=99:
        temp = '000'+str(i)
    elif i <=999:
        temp = '00'+str(i)
    elif i <=9999:
        temp = '0'+str(i)
    for j in pic_list:
        if str(temp) in j:
            kk = kk+1
            print kk
            shutil.copy(osp.join('./jpg/',j),osp.join('./train/',str(label['labels'][0][i-1]-1)))
            break

kk = 0
for i in data['tstid'][0]:
    if i<=9:
        temp = '0000'+str(i)
    elif i <=99:
        temp = '000'+str(i)
    elif i <=999:
        temp = '00'+str(i)
    elif i <=9999:
        temp = '0'+str(i)
    for j in pic_list:
        if str(temp) in j:
            kk = kk+1
            print kk
            shutil.copy(osp.join('./jpg/',j),osp.join('./test/',str(label['labels'][0][i-1]-1)))
            break