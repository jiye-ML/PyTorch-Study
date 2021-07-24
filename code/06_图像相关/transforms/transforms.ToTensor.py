"""
CLASStorchvision.transforms.ToTensor[SOURCE]
Convert a PIL Image or numpy.ndarray to tensor.

Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8

In the other cases, tensors are returned without scaling.

__call__(pic)[SOURCE]
Parameters
pic (PIL Image or numpy.ndarray) – Image to be converted to tensor.

Returns
Converted image.

Return type
Tensor
"""