"""

torchvision.transforms.ToPILImage(mode=None)[SOURCE]
Convert a tensor or an ndarray to PIL Image.

Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image while preserving the value range.

Parameters
mode (PIL.Image mode) –

color space and pixel depth of input data (optional). If mode is None (default) there are some assumptions made about the input data:

If the input has 4 channels, the mode is assumed to be RGBA.

If the input has 3 channels, the mode is assumed to be RGB.

If the input has 2 channels, the mode is assumed to be LA.

If the input has 1 channel, the mode is determined by the data type (i.e int, float, short).

__call__(pic)[SOURCE]
Parameters
pic (Tensor or numpy.ndarray) – Image to be converted to PIL Image.

Returns
Image converted to PIL Image.

Return type
PIL Image

"""