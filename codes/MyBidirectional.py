#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/1 20:34
@File:          MyBidirectional.py
'''

import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

class MyBidirectional(Layer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer, **args):
        super(MyBidirectional, self).__init__(**args)
        self.supports_masking = True
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name

    def reverse_sequence(self, x, mask):
        seq_len = K.sum(K.cast(mask, 'int32'), axis=1)
        return tf.reverse_sequence(x, seq_len, seq_dim=1)

    def call(self, inputs, mask=None, **kwargs):
        x_forward = self.forward_layer(inputs)
        x_backward = self.reverse_sequence(inputs, mask)
        x_backward = self.backward_layer(x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], -1)

        if K.ndim(x) == 3:
            x = x * K.expand_dims(K.cast(mask, K.dtype(x)), axis=2)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.forward_layer.units * 2,)

    def compute_mask(self, inputs, mask=None):
        return mask