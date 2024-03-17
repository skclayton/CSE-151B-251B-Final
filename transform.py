from loadData import *

input_transform = standard_transforms.Compose([
        standard_transforms.RandomRotation(degrees = 10),
        standard_transforms.RandomHorizontalFlip(p=1),
        standard_transforms.RandomVerticalFlip(p=1), 
        standard_transforms.ToTensor()
    ])

