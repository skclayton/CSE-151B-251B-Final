from loadData import *

input_transform = standard_transforms.Compose([
        standard_transforms.RandomRotation(degrees = 10),
        standard_transforms.RandomHorizontalFlip(p=1),
        standard_transforms.RandomVerticalFlip(p=1), 
        standard_transforms.ToTensor(),
        standard_transforms.RandomResizedCrop(size=(128, 128), scale=(0.5, 0.75))
    ])

