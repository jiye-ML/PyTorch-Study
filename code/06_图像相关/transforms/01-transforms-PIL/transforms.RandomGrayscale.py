''''
torchvision.transforms.RandomGrayscale(p=0.1)[SOURCE]
Randomly convert image to grayscale with a probability of p (default 0.1).

Parameters
p (float) – probability that image should be converted to grayscale.

Returns
Grayscale version of the input image with probability p and unchanged with probability (1-p). - If input image is 1 channel: grayscale version is 1 channel - If input image is 3 channel: grayscale version is 3 channel with r == g == b

Return type
PIL Image

'''