import numpy as np
import random
import torch
# random.seed(7)
# print(random.uniform(0,1))
# print(7*7*9)
# exit()

"""取出各个cell中特定channl 的box"""
x = torch.Tensor(np.arange(441).reshape(9,7,7,1))
y = torch.randn(9,7,7,4)
print(x)
print(x.shape)
max,idx = torch.max(x,dim=0)
print("=====")
# print(max.shape)
print(max)
print(idx)

i = 0
idx = idx.view(49)
boxes =[]
for h in range(7):
    for w in range(7):
        box = y[idx[i]][h][w][:]
        boxes.append(box)
        i+=1
print(boxes)
boxes = np.stack(boxes)
boxes = np.reshape((boxes,(7,7,4)))
print(boxes.shape)
print(boxes)
#
# y_max = y
# x_max = x[idx]
# print(x_max)
# print(x_max.shape)