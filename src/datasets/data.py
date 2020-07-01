import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as tvtf


class data_raw(data.Dataset):
    def __init__(self, data_root_dir, train_dir, val_dir, annotation_train_dir, annotation_val_dir, is_train=True):
        super().__init__()

        self.is_train = is_train
        self.root_dir = data_root_dir

        self.train_dir = os.path.join(self.root_dir, train_dir)
        self.val_dir = os.path.join(self.root_dir, val_dir)


        self.data = None
        self.annotations = None

        if self.is_train:   
            self.annotation_dir = os.path.join(self.root_dir, annotation_train_dir)
            self.data = os.listdir(self.train_dir)
            self.annotations = os.listdir(self.annotation_dir)
        elif self.is_train == False:        
            self.annotation_dir = os.path.join(self.root_dir, annotation_val_dir)
            self.data = os.listdir(self.val_dir)
            self.annotations = os.listdir(self.annotation_dir)

        self.data.sort()
        self.annotations.sort()

    def __getitem__(self, index):
        if self.is_train:
            item_path = self.train_dir + self.data[index]
        else:
            item_path = self.val_dir + self.data[index]
        
        annotation_path = self.annotation_dir + self.annotations[index]
        image = Image.open(item_path).convert('RGB')
        annotation = Image.open(annotation_path).convert('LA')


        tf = tvtf.Compose([tvtf.Resize(224),
                           tvtf.CenterCrop(224),
                           tvtf.ToTensor(),
                           tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                           ])
        
        tf_label = tvtf.Compose([tvtf.Resize(224),
                           tvtf.CenterCrop(224),
                           tvtf.ToTensor()
                           ])
        image, annotation = tf(image), tf_label(annotation)
        return (image, annotation)

    def __len__(self):
        return len(self.data)


# test
if __name__ == "__main__":
    dataset = data_raw('/home/ken/ppnckh/src/data/', 'train_image/','val_image/',
                         'train_segmentation/','val_segmentation/',is_train=False)
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(torch.max(dataset[0][0]).item())
    print(torch.max(dataset[0][1]).item())


