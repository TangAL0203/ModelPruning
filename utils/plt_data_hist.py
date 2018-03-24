#-*-coding:utf-8-*-
import numpy as np 
import matplotlib.pyplot as plt 


path = './new_img_name.txt'
f = open(path)
name_list = f.readlines()
f.close()
# num_per_type = {str(i):0 for i in range(1,103,1)}object, dtype, copy, order, subok, ndmin
num_per_type = []

for line in name_list:
    # num_per_type[line.split('-')[0]] +=1
    num_per_type.append(int(line.split('-')[0]))

num_per_type = np.array(num_per_type)
plt.hist(num_per_type, bins=100, color='steelblue')
plt.show()
