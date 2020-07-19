# code 1
# 仅供测试基本原理，列出公式
import numpy as np
import torch
import torchvision

# a example of 2-layer neural network
N, D_IN, H, D_OUT=64,1000,100,10
x=torch.randn(N,D_IN).cuda()
y=torch.randn(N,D_OUT).cuda()
w1=torch.randn(D_IN,H,requires_grad=True).cuda()
w2=torch.randn(H,D_OUT,requires_grad=True).cuda()

learning_rate=1e-6

for it in range(500):
    y_pred=x.mm(w1).clamp(min=0).mm(w2)

    loss=(y_pred-y).pow(2).sum()
    print(it,loss.item())

    grad_w2=x.mm(w1).T.clamp(min=0).mm(2*(y_pred-y))
    grad_w1=x.T.mm(2*(y_pred-y)).mm(w2.T)
    
#    loss.backward()

#    with torch.no_grad():
#        w1 -=learning_rate * w1.grad
#        w2 -=learning_rate * w2.grad
#        w1.grad.zero_()
#        w2.grad.zero_()
    w1 -=learning_rate * grad_w1
    w2 -=learning_rate * grad_w2
