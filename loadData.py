from torch.utils import data
import os
from os.path import isfile, join
from PIL import Image
import numpy as np
import torchvision.transforms as standard_transforms

def make_dataset(task, mode):
    items = []
    
    if task == 'prediction': 
        root = '/home/esong1/age_prediction'
        base_path = join(root, mode)
        
        for age in range(1, 101):
            path = join(base_path, format(age, '03d'))
            images = [(join(path, f), age) for f in os.listdir(path) if isfile(join(path, f))]
            items.extend(images)
            
    return items

class Dataset(data.Dataset):
    def __init__(self, args, mode):
        self.task = args.task
            
        self.imgs = make_dataset(self.task, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.width = 128
        self.height = 128

        self.transform = transform
        
    def __getitem__(self, index):
        if self.task == 'prediction':
            img_path, transform = self.imgs[index]
            label = self.labels[index]
            img = Image.open(img_path).convert('RGB').resize((self.width, self.height))

            return np.array(img), label
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.imgs) 
    #default





    def __getitem__(self, index):
        img_path, transform = self.img_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB').resize((self.width, self.height))
        

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    

