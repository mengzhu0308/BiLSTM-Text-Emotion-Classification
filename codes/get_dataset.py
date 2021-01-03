#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/13 12:31
@File:          dataset.py
'''

import pandas as pd
import numpy as np
from jieba import posseg as pseg

def get_dataset(dataset_dir='D:/datasets/text_classification', stop_flag=('x', 'c', 'u', 'd', 'p', 't', 'uj', 'f', 'r')):
    train_dataset = pd.read_excel(f'{dataset_dir}/train.xlsx', usecols=['Class Index', 'Description'], engine='openpyxl')
    val_dataset = pd.read_excel(f'{dataset_dir}/val.xlsx', usecols=['Class Index', 'Description'], engine='openpyxl')

    X_train, Y_train = pd.Series(dtype=object), pd.Series(dtype='int32')
    X_val, Y_val = pd.Series(dtype=object), pd.Series(dtype='int32')

    cnt = 0
    sample_len = len(train_dataset['Class Index'].values)
    for i in range(sample_len):
        text = train_dataset['Description'].loc[i]
        cls_id = train_dataset['Class Index'].loc[i]

        if text is None or not isinstance(text, str) or text == '':
            break

        if cls_id is None or not isinstance(cls_id, np.int64):
            break

        text_seged = pseg.cut(text.lower())

        term_list = []
        for term, flag in text_seged:
            if flag not in stop_flag:
                term_list.append(term) if not term.isdigit() else term_list.append('num')

        X_train.loc[cnt] = term_list
        Y_train.loc[cnt] = cls_id - 1
        cnt += 1

    cnt = 0
    sample_len = len(val_dataset['Class Index'].values)
    for i in range(sample_len):
        text = val_dataset['Description'].loc[i]
        cls_id = val_dataset['Class Index'].loc[i]

        if text is None or not isinstance(text, str) or text == '':
            break

        if cls_id is None or not isinstance(cls_id, np.int64):
            break

        text_seged = pseg.cut(text.lower())

        term_list = []
        for term, flag in text_seged:
            if flag not in stop_flag:
                term_list.append(term) if not term.isdigit() else term_list.append('num')

        X_val.loc[cnt] = term_list
        Y_val.loc[cnt] = cls_id - 1
        cnt += 1

    return (X_train, Y_train), (X_val, Y_val)
