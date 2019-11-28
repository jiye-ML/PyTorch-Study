'''


CLASStorchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None, fill=0)[SOURCE]
Rotate the image by angle.

Parameters
degrees (sequence or float or int) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).

resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional) – An optional resampling filter. See filters for more information. If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.

expand (bool, optional) – Optional expansion flag. If true, expands the output to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.

center (2-tuple, optional) – Optional center of rotation. Origin is the upper left corner. Default is the center of the image.

fill (3-tuple or int) – RGB pixel fill value for area outside the rotated image. If int, it is used for all channels respectively.

'''