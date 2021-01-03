#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/29 18:45
@File:          BiLSTM_model.py
'''

from keras.layers import *

from MyBidirectional import MyBidirectional

def BiLSTM_Model(x, vocab_size, hidden_dim, num_classes=4):
    x = Embedding(vocab_size, hidden_dim, mask_zero=True)(x)
    x = MyBidirectional(LSTM(hidden_dim, return_sequences=True, activation='relu'))(x)
    x = MyBidirectional(LSTM(hidden_dim * 2, return_sequences=True, activation='relu'))(x)
    x = MyBidirectional(LSTM(hidden_dim * 4, return_sequences=True, activation='relu'))(x)
    x = MyBidirectional(LSTM(hidden_dim * 8, activation='relu'))(x)
    x = Dense(num_classes)(x)

    return x