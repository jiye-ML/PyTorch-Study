import torch
import time


x  = torch.rand([3,3], device='cuda')
x_ = x.cpu().numpy()

y  = torch.rand([3,3], requires_grad=True, device='cuda')
y_ = y.cpu().detach().numpy()
# y_ = y.detach().cpu().numpy() 也可以
# 二者好像差别不大？我们来比比时间：
start_t = time.time()
for i in range(10000):
    y_ = y.cpu().detach().numpy()
print(time.time() - start_t)
# 1.1049120426177979

start_t = time.time()
for i in range(10000):
    y_ = y.detach().cpu().numpy()
print(time.time() - start_t)
# 1.115112543106079
# 时间差别不是很大，当然，这个速度差别可能和电脑配置
# （比如 GPU 很贵，CPU 却很烂）有关。