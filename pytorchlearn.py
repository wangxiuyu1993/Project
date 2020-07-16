import numpy as np
import torch
import torchvision

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


    loss.backward()
    print(type(w1.grad))

    with torch.no_grad():
        w1 -=learning_rate * w1.grad
        w2 -=learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()