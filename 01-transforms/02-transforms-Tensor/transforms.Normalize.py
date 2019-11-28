"""
CLASStorchvision.transforms.Normalize(mean, std, inplace=False)[SOURCE]
Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel of the input torch.*Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]

NOTE

This transform acts out of place, i.e., it does not mutates the input tensor.

Parameters
mean (sequence) – Sequence of means for each channel.

std (sequence) – Sequence of standard deviations for each channel.

inplace (bool,optional) – Bool to make this operation in-place.

__call__(tensor)[SOURCE]
Parameters
tensor (Tensor) – Tensor image of size (C, H, W) to be normalized.

Returns
Normalized Tensor image.

Return type
Tensor
"""