
import torch
import torch.nn as nn
import time

N,D_IN,H,D_OUT=64,1000,100,10


t0=time.process_time()
# 初始化
x=torch.randn(N,D_IN).cuda()
y=torch.randn(N,D_OUT).cuda()

# 设定model
model=nn.Sequential(nn.Linear(D_IN,H), nn.ReLU(), nn.Linear(H,D_OUT))
model.cuda()

# 设定loss fun
loss_fn=nn.MSELoss(reduction='sum')
learning_rate=1e-6

for it in range(500):
    y_pred=model(x)
    loss=loss_fn(input=y_pred,target=y)
#    print(it,loss.item())

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -=learning_rate* param.grad

t1=time.process_time()
print("运行时间是：{}s".format(t1-t0))





    

