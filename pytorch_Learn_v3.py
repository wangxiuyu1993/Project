import torch.nn as nn
import torch
import numpy as np

NUM_DIGIT=10
NUM_HIDDEN=100

# 定义fizzbuzz游戏
def fiz_buz_encode(i):
    if i%15==0:
        return 3
    elif i%5==0:
        return 2
    elif i%3==0:
        return 1
    else:
        return 0

def fiz_buz_decode(i,prediction):
    return [str(i),"fizz","buzz","fizzbuzz"][prediction]

def helper(i):
    print(fiz_buz_decode(i,fiz_buz_encode(i)))

# 定义转化为二进制的函数，声明训练集合
def binary_encode(i,NUM_DIGIT):
    return np.array([i >> d & 1 for d in range(NUM_DIGIT)][::-1])

# 初始化训练集
Trx=torch.tensor([binary_encode(i,NUM_DIGIT) for i in range(101,2**NUM_DIGIT)])
Try=torch.tensor([fiz_buz_encode(i) for i in range(101, 2**NUM_DIGIT)])

print(Trx.shape)
print(Try.shape)

model=nn.Sequential(nn.Linear(NUM_DIGIT,NUM_HIDDEN),nn.ReLU(),nn.Linear(NUM_HIDDEN,4))
loss_fn=nn.CrossEntropyLoss()
opitimizer=torch.optim.SGD(model.parameters(),lr=0.05)

