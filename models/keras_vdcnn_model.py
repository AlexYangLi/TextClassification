# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_vdcnn_model.py

@time: 2019/2/8 13:01

@desc:

"""

import math
from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, Flatten, Dense, BatchNormalization, ReLU, Add, \
    MaxPooling1D
from keras import backend as K

from models.keras_base_model import KerasBaseModel
from layers.kmaxpooling import KMaxPooling


class VDCNN(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(VDCNN, self).__init__(config, **kwargs)

    def build(self, depth=[4, 4, 10, 10], pooling_type='maxpool', use_shortcut = False):
        input_text = Input(shape=(self.max_len,))

        embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                    weights=[self.word_embeddings],
                                    trainable=self.config.word_embed_trainable)(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        # first temporal conv layer
        conv_out = Conv1D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding='same')(text_embed)
        shortcut = conv_out

        # temporal conv block: 64
        for i in range(depth[0]):
            if i < depth[0] - 1:
                shortcut = conv_out
                conv_out = self.conv_block(inputs=conv_out, filters=64, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = self.conv_block(inputs=conv_out, filters=64, use_shortcut=use_shortcut, shortcut=None)

        # down-sampling
        # shortcut is the second last conv block output
        conv_out = self.dowm_sampling(inputs=conv_out, pooling_type=pooling_type, use_shortcut=use_shortcut,
                                      shortcut=shortcut)
        shortcut = conv_out

        # temporal conv block: 128
        for i in range(depth[1]):
            if i < depth[1] - 1:
                shortcut = conv_out
                conv_out = self.conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = self.conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=None)

        # down-sampling
        conv_out = self.dowm_sampling(inputs=conv_out, pooling_type=pooling_type, use_shortcut=use_shortcut,
                                      shortcut=shortcut)
        shortcut = conv_out

        # temporal conv block: 256
        for i in range(depth[2]):
            if i < depth[1] - 1:
                shortcut = conv_out
                conv_out = self.conv_block(inputs=conv_out, filters=256, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = self.conv_block(inputs=conv_out, filters=256, use_shortcut=use_shortcut, shortcut=None)

        # down-sampling
        conv_out = self.dowm_sampling(inputs=conv_out, pooling_type=pooling_type, use_shortcut=use_shortcut,
                                      shortcut=shortcut)

        # temporal conv block: 512
        for i in range(depth[3]):
            if i < depth[1] - 1:
                shortcut = conv_out
                conv_out = self.conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=shortcut)
            else:
                # shortcut is not used at the last conv block
                conv_out = self.conv_block(inputs=conv_out, filters=128, use_shortcut=use_shortcut, shortcut=None)

        # 8-max pooling
        conv_out = KMaxPooling(k=8)(conv_out)
        flatten = Flatten()(conv_out)

        fc1 = Dense(2048, activation='relu')(flatten)
        sentence_embed = Dense(2048, activation='relu')(fc1)

        dense_layer = Dense(256, activation='relu')(sentence_embed)
        output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model

    def conv_block(self, inputs, filters, use_shortcut, shortcut):
        conv_1 = Conv1D(filters=filters, kernel_size=3, kernel_initializer='he_uniform', padding='same')(inputs)
        bn_1 = BatchNormalization()(conv_1)
        relu_1 = ReLU()(bn_1)
        conv_2 = Conv1D(filters=filters, kernel_size=3, kernel_initializer='he_uniform', padding='same')(relu_1)
        bn_2 = BatchNormalization()(conv_2)
        relu_2 = ReLU()(bn_2)

        if shortcut is not None and use_shortcut:
            return Add()([inputs, shortcut])
        else:
            return relu_2

    def dowm_sampling(self, inputs, pooling_type, use_shortcut, shortcut):
        if pooling_type == 'kmaxpool':
            k = math.ceil(K.int_shape(inputs)[1] / 2)
            pool = KMaxPooling(k)(inputs)
        elif pooling_type == 'maxpool':
            pool = MaxPooling1D(pool_size=3, strides=2, padding='same')(inputs)
        elif pooling_type == 'conv':
            pool = Conv1D(filters=K.int_shape(inputs)[-1], kernel_size=3, strides=2,
                          kernel_initializer='he_uniform', padding='same')(inputs)
        else:
            raise ValueError('pooling_type `{}` not understood'.format(pooling_type))
        if shortcut is not None and use_shortcut:
            shortcut = Conv1D(filters=K.int_shape(inputs)[-1], kernel_size=3, strides=2,
                              kernel_initializer='he_uniform', padding='same')(shortcut)
            return Add()([pool, shortcut])
        else:
            return pool
