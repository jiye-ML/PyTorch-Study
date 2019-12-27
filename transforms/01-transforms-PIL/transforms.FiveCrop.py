'''
https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.FiveCrop



torchvision.transforms.FiveCrop(size)[SOURCE]
Crop the given PIL Image into four corners and the central crop

NOTE

This transform returns a tuple of images and there may be a mismatch in the number of inputs and targets your Dataset returns. See below for an example of how to deal with this.

Parameters
size (sequence or int) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop of size (size, size) is made.

Example


'''

transform = Compose([FiveCrop(size), # this is a list of PIL Images
                  Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
])
#In your test loop you can do the following:
input, target = batch # input is a 5d tensor, target is 2d
bs, ncrops, c, h, w = input.size()
result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops