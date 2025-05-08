import cv2
import os
import random
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset

class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir, file_path, transform = None, image_size = 256):
        super().__init__()  
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size

        with open(file_path, "r") as f:
            lines = f.read()
            self.image_list = [[i.split(' ')[0], i.split(' ')[1]] for i in lines.split('\n')]
            random.shuffle(self.image_list)

            print('init:',len(self.image_list))
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):

        img_path, label = self.image_list[index]
        img_path = os.path.join(self.root_dir, img_path)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            return None
        
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(img) == 0:
            return {'image': None, 'label': None ,'filename': img_path}
        
        img = self.transform(img)
        label = int(label)
        label = torch.tensor(label)
            
        return {'image': img, 'label': label, 'filename': img_path}