#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/16 6:51
@File:          Dataset.py
'''

class Dataset:
    def __init__(self, texts, labels, text_transform=None, label_transform=None):
        self.texts = texts
        self.labels = labels
        self.__text_transform = text_transform
        self.__label_transform = label_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        if self.__text_transform is not None:
            text = self.__text_transform(text)
        if self.__label_transform is not None:
            label = self.__label_transform(label)

        return text, label