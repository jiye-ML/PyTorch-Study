import torch
import torch.nn as nn
import torch.nn.modules.batchnorm


# 创建随机输入
def create_inputs():
    return torch.randn(8, 3, 20, 20)


# 以 BatchNorm2d 为例
# mean_val, var_val 不为None时，不对输入进行统计，而直接用传进来的均值、方差
def dummy_bn_forward(x, bn_weight, bn_bias, eps, mean_val=None, var_val=None):
    if mean_val is None:
        mean_val = x.mean([0, 2, 3])
    if var_val is None:
        # 这里需要注意，torch.var 默认算无偏估计，因此需要手动设置unbiased=False
        var_val = x.var([0, 2, 3], unbiased=False)

    x = x - mean_val[None, ..., None, None]
    x = x / torch.sqrt(var_val[None, ..., None, None] + eps)
    x = x * bn_weight[..., None, None] + bn_bias[..., None, None]
    return mean_val, var_val, x


running_mean = torch.zeros(3)
running_var = torch.ones_like(running_mean)
momentum = 0.1  # 这也是BN初始化时momentum默认值
bn_layer = nn.BatchNorm2d(num_features=3, momentum=momentum)

# 模拟 forward 10 次
for t in range(10):
    inputs = create_inputs()
    bn_outputs = bn_layer(inputs)
    inputs_mean, inputs_var, _ = dummy_bn_forward(
        inputs, bn_layer.weight, bn_layer.bias, bn_layer.eps
    )
    n = inputs.numel() / inputs.size(1)
    # 更新 running_var 和 running_mean
    running_var = running_var * (1 - momentum) + momentum * inputs_var * n / (n - 1)
    running_mean = running_mean * (1 - momentum) + momentum * inputs_mean

assert torch.allclose(running_var, bn_layer.running_var)
assert torch.allclose(running_mean, bn_layer.running_mean)
print(f'bn_layer running_mean is {bn_layer.running_mean}')
print(f'dummy bn running_mean is {running_mean}')
print(f'bn_layer running_var is {bn_layer.running_var}')
print(f'dummy bn running_var is {running_var}')

""" output
bn_layer running_mean is tensor([-0.0011, -0.0017,  0.0004])
dummy bn running_mean is tensor([-0.0011, -0.0017,  0.0004])
bn_layer running_var is tensor([0.9937, 1.0174, 1.0047])
dummy bn running_var is tensor([0.9937, 1.0174, 1.0047])
"""
