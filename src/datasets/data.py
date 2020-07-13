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
            self.annotation_dir = os.path.join(
                self.root_dir, annotation_train_dir)
            self.data = os.listdir(self.train_dir)
            self.annotations = os.listdir(self.annotation_dir)
        elif self.is_train == False:
            self.annotation_dir = os.path.join(
                self.root_dir, annotation_val_dir)
            self.data = os.listdir(self.val_dir)
            self.annotations = os.listdir(self.annotation_dir)

        self.data.sort()
        self.annotations.sort()
        self.data = self.data
        self.annotations = self.annotations

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
        annotation = annotation[0, :, :]
        return (image, annotation.long())

    def __len__(self):
        return len(self.data)


# test

class data_pose(data.Dataset):
    def __init__(self, data_root_dir, train_dir, val_dir, train_pose_dir, val_pose_dir, annotation_train_dir, annotation_val_dir, is_train=True):
        super().__init__()

        self.is_train = is_train
        self.root_dir = data_root_dir

        self.train_dir = os.path.join(self.root_dir, train_dir)
        self.train_pose_dir = os.path.join(self.root_dir, train_pose_dir)

        self.val_dir = os.path.join(self.root_dir, val_dir)
        self.val_pose_dir = os.path.join(self.root_dir, val_pose_dir)

        self.data = None
        self.annotations = None

        if self.is_train:
            self.annotation_dir = os.path.join(
                self.root_dir, annotation_train_dir)
            self.data = os.listdir(self.train_dir)
            self.data_pose = os.listdir(self.train_pose_dir)
            self.annotations = os.listdir(self.annotation_dir)
        elif self.is_train == False:
            self.annotation_dir = os.path.join(
                self.root_dir, annotation_val_dir)
            self.data = os.listdir(self.val_dir)
            self.data_pose = os.listdir(self.val_pose_dir)
            self.annotations = os.listdir(self.annotation_dir)

        self.data.sort()
        self.data_pose = [i for i in self.data_pose if i.endswith('_IUV.png')]
        self.data_pose.sort()
        self.annotations.sort()
        # header
        self.data = self.data[:5]
        self.data_pose = self.data_pose[:5]
        self.annotations = self.annotations[:5]

    def __getitem__(self, index):
        if self.is_train:
            item_path = self.train_dir + self.data[index]
            pose_path = self.train_pose_dir + self.data_pose[index]
        else:
            item_path = self.val_dir + self.data[index]
            pose_path = self.val_pose_dir + self.data_pose[index]
        # print(pose_path)
        annotation_path = self.annotation_dir + self.annotations[index]
        image = Image.open(item_path).convert('RGB')
        pose = Image.open(pose_path).convert('L')
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
        image, pose, annotation = tf(image),\
            tf_label(pose), tf_label(annotation)

        annotation = annotation[0, :, :]
        image = torch.cat((image, pose), 0)
        return (image, annotation.long())

    def __len__(self):
        return len(self.data)


# test
if __name__ == "__main__":
    # raw dataset
    # dataset = data_raw('/home/ken/ppnckh/ppnckh/src/data/', 'train_image/', 'val_image/',
    #                    'train_segmentation/', 'val_segmentation/', is_train=False)
    # pose dataset
    dataset = data_pose('/home/ken/ppnckh/ppnckh/src/data/', 'train_image/', 'val_image/',
                        'train_pose/', 'val_pose/', 'train_segmentation/', 'val_segmentation/', is_train=True)
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(torch.max(dataset[0][0]).item())
    print(torch.max(dataset[0][1]).item())
