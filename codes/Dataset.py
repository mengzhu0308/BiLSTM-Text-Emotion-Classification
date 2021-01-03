#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/16 6:51
@File:          Dataset.py
'''

class Dataset:
    def __init__(self, images, labels, image_transform=None, label_transform=None):
        self.images = images
        self.labels = labels
        self.__image_transform = image_transform
        self.__label_transform = label_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        if self.__image_transform is not None:
            image = self.__image_transform(image)
        if self.__label_transform is not None:
            label = self.__label_transform(label)

        return image, label