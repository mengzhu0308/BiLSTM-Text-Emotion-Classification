#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/13 17:00
@File:          utils.py
'''

import numpy as np

def str2id(text, tokenid):
    out_text = []
    for term in text:
        out_text.append(tokenid[term])
    return out_text

def sequence_padding(text_list, max_length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if max_length is None:
        max_length = max([len(text) for text in text_list])

    outputs = []

    for text in text_list:
        text = text[:max_length]
        pad_width = (0, max_length - len(text))
        text = np.pad(text, pad_width, mode='constant', constant_values=padding)
        outputs.append(text)

    return np.array(outputs, dtype='int32')