"""
模拟 BN 的前向操作
"""
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

bn_layer = nn.BatchNorm2d(num_features=3)
inputs = create_inputs()
# 用 pytorch 的实现 forward
bn_outputs = bn_layer(inputs)
# 用 dummy bn 来 forward
_, _, expected_outputs = dummy_bn_forward(
    inputs, bn_layer.weight, bn_layer.bias, bn_layer.eps)
assert torch.allclose(expected_outputs, bn_outputs)
