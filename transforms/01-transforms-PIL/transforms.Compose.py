'''
CLASStorchvision.transforms.Compose(transforms)[SOURCE]
Composes several transforms together.

Parameters
transforms (list of Transform objects) – list of transforms to compose.

Example

>>> transforms.Compose([
>>>     transforms.CenterCrop(10),
>>>     transforms.ToTensor(),
>>> ])
'''