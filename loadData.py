from torch.utils import data
import os
from os.path import isfile, join
from PIL import Image
import numpy as np
import torchvision.transforms as standard_transforms
from transform import *

def make_dataset(task, mode):
    items = []
    root = '/home/esong1/age_prediction'
    transform = input_transform()

    for class_folder in os.listdir(root):
        class_path = os.path.join(root, class_folder)

        if os.path.isdir(class_path):
            files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            apply_transform = len(files) < 1000  

            for f in files:
                item_path = os.path.join(class_path, f)
                item = (item_path, class_folder)  

                if apply_transform:
                    item = (item_path, class_folder, transform)

                items.append(item)


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
        self.mode = mode  
        self.imgs = make_dataset(self.task, self.mode)
        
        # self.width = 128
        # self.height = 128

        self.transform = input_transform
        
    def __getitem__(self, index):
        # if self.task == 'prediction':
        #     img_path, transform = self.imgs[index]
        #     label = self.labels[index]
        #     img = Image.open(img_path).convert('RGB').resize((self.width, self.height))

        #     return np.array(img), label
        
        # if self.transform is not None:
        #     image = self.transform(image)

        img_path, label = self.imgs[index]  

        img = Image.open(img_path).convert('RGB')  
        if self.transform:
            img = self.transform(img)  

        return img, label


    def __len__(self):
        return len(self.imgs) 





    # def __getitem__(self, index):
    #     img_path, transform = self.img_paths[index]
    #     label = self.labels[index]
    #     image = Image.open(img_path).convert('RGB').resize((self.width, self.height))
        

    #     if self.transform is not None:
    #         image = self.transform(image)
        
    #     return image, label
    

