import torch.nn as nn
import torch
import numpy as np

NUM_DIGIT=10
NUM_HIDDEN=100
BATCH_SIZE=128

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
Trx=torch.tensor([binary_encode(i,NUM_DIGIT) for i in range(101,2**NUM_DIGIT)],dtype=torch.float)
Try=torch.LongTensor([fiz_buz_encode(i) for i in range(101, 2**NUM_DIGIT)])

print(Trx.shape)
print(Try.shape)

model=nn.Sequential(nn.Linear(NUM_DIGIT,NUM_HIDDEN),nn.ReLU(),nn.Linear(NUM_HIDDEN,4))
loss_fn=nn.CrossEntropyLoss()
opitimizer=torch.optim.Adam(model.parameters(),lr=0.05)

if torch.cuda.is_available():
    model=model.cuda()
    Trx=Trx.cuda()
    Try=Try.cuda()

for epoch in range(1,10000):
    for start in range(0,len(Trx),BATCH_SIZE):
        end=start+BATCH_SIZE
        batchX=Trx[start:end]
        batchY=Try[start:end]

        y_pred=model(batchX)

        loss=loss_fn(y_pred,batchY)

        print("EPOCH",epoch, loss.item())

        opitimizer.zero_grad()

        loss.backward()
        opitimizer.step()

