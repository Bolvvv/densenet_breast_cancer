import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, filename, image_dir, transform):
        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.transform = transform
    
    def __getitem__(self, i):
        index = i % self.len
        image_name , label = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        img = Image.open(image_path)

        #对图片进行预处理
        img=self.transform(img)
        
        return img, int(label)
    
    def __len__(self):
        data_len = len(self.image_label_list)
        return data_len

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split()
                name = content[0]
                label = content[1]
                image_label_list.append((name, label))
        return image_label_list

                