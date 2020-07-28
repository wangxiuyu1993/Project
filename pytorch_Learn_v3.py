import torch.nn as nn
import torch
import numpy as np
import time

start=time.process_time()

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
# x：从101到1024，转化为二进制
# y：输出 1 2 3，即对应fizbuz游戏输出
Trx=torch.tensor([binary_encode(i,NUM_DIGIT) for i in range(101,2**NUM_DIGIT)],dtype=torch.float)
Try=torch.LongTensor([fiz_buz_encode(i) for i in range(101, 2**NUM_DIGIT)])

print(Trx.shape)
print(Try.shape)

model=nn.Sequential(nn.Linear(NUM_DIGIT,NUM_HIDDEN),nn.ReLU(),nn.Linear(NUM_HIDDEN,4))
loss_fn=nn.CrossEntropyLoss()
opitimizer=torch.optim.Adam(model.parameters(),lr=0.01)

if torch.cuda.is_available():
    model=model.cuda()
    Trx=Trx.cuda()
    Try=Try.cuda()

for epoch in range(1,2000):
    for start in range(0,len(Trx),BATCH_SIZE):
        # 训练集合：每次拿出BATCH数量的样本进行训练
        end=start+BATCH_SIZE
        batchX=Trx[start:end]
        batchY=Try[start:end]

        y_pred=model(batchX)

        loss=loss_fn(y_pred,batchY)

        print("EPOCH",epoch, loss.item())

        opitimizer.zero_grad()

        loss.backward()
        opitimizer.step()

testX=torch.tensor([binary_encode(i,NUM_DIGIT) for i in range(1,100)],dtype=torch.float)
if torch.cuda.is_available()==True:
    testX=testX.cuda()
# 这么写是因为这个tensor不需要grad，可以避免爆内存 
with torch.no_grad():
    testY=model(testX)
predictions=zip(range(1,100),testY.max(1)[1].data.tolist())

print([fiz_buz_decode(i,x) for i,x in predictions])
end=time.process_time()
print('time is %6.3f'%(end-start))