#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/1 20:12
@File:          train.py
'''

import math
import numpy as np
import pandas as pd
from gensim import corpora
from keras.layers import Input
from keras import Model
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras import backend as K

from Dataset import Dataset
from get_dataset import get_dataset
from generator import generator
from utils import str2id, sequence_padding
from Loss import Loss
from ToOneHot import ToOneHot
from BiLSTM_model import BiLSTM_Model

class CrossEntropy(Loss):
    def compute_loss(self, inputs):
        y_true, y_pred = inputs
        loss = K.categorical_crossentropy(y_true, K.softmax(y_pred))
        return K.mean(loss)

if __name__ == '__main__':
    num_classes = 4
    vocab_size = 33106
    max_length = 64
    hidden_dim = 64
    train_batch_size = 128
    val_batch_size = 500

    (X_train, Y_train), (X_val, Y_val) = get_dataset()
    dictionary = corpora.Dictionary(pd.concat([X_train, X_val]))

    X_train = [str2id(x, dictionary.token2id) for x in X_train]
    X_val = [str2id(x, dictionary.token2id) for x in X_val]

    X_train = sequence_padding(X_train, max_length=max_length)
    Y_train = np.array(Y_train, dtype='int32')
    X_val = sequence_padding(X_val, max_length=max_length)
    Y_val = np.array(Y_val, dtype='int32')

    train_dataset = Dataset(X_train, Y_train, label_transform=ToOneHot(num_classes))
    val_dataset = Dataset(X_val, Y_val, label_transform=ToOneHot(num_classes))
    train_generator = generator(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_generator = generator(val_dataset, batch_size=val_batch_size, shuffle=False)

    text_input = Input(shape=(None, ), name='text_input', dtype='int32')
    y_true = Input(shape=(num_classes, ), dtype='int32')
    out = BiLSTM_Model(text_input, vocab_size, hidden_dim, num_classes=num_classes)
    out = CrossEntropy(-1)([y_true, out])
    model = Model([y_true, text_input], out)
    opt = Adam()
    model.compile(opt)

    num_train_batches = math.ceil(len(Y_train) / train_batch_size)
    num_val_examples = len(Y_val)
    num_val_batches = math.ceil(num_val_examples / val_batch_size)

    def evaluate(model):
        total_loss = 0.
        total_corrects = 0

        for _ in range(num_val_batches):
            batch_data, _ = next(val_generator)
            val_loss, predict = model.test_on_batch(batch_data, y=None), model.predict_on_batch(batch_data)

            total_loss += val_loss
            total_corrects += np.sum(np.argmax(batch_data[0], axis=-1) == np.argmax(predict, axis=-1))

        val_loss = total_loss / num_val_batches
        val_acc = (total_corrects / num_val_examples) * 100

        return val_loss, val_acc

    class Evaluator(Callback):
        def __init__(self):
            super(Evaluator, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            val_loss, val_acc = evaluate(self.model)

            print(f'val_loss = {val_loss:.5f}, val_acc = {val_acc:.2f}')

    evaluator = Evaluator()

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_batches,
        epochs=10,
        callbacks=[evaluator],
        shuffle=False,
        initial_epoch=0
    )
