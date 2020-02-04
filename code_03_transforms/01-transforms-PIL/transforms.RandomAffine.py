'''
CLASStorchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)[SOURCE]
Random affine transformation of the image keeping center invariant

Parameters
degrees (sequence or float or int) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees). Set to 0 to deactivate rotations.

translate (tuple, optional) – tuple of maximum absolute fraction for horizontal and vertical translations. For example translate=(a, b), then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.

scale (tuple, optional) – scaling factor interval, e.g (a, b), then scale is randomly sampled from the range a <= scale <= b. Will keep original scale by default.

shear (sequence or float or int, optional) – Range of degrees to select from. If shear is a number, a shear parallel to the x axis in the range (-shear, +shear) will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values, a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied. Will not apply shear by default

resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional) – An optional resampling filter. See filters for more information. If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.

fillcolor (tuple or int) – Optional fill color (Tuple for RGB Image And int for grayscale) for the area outside the transform in the output image.(Pillow>=5.0.0)
'''