# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_multi_text_cnn_model.py

@time: 2019/2/8 10:55

@desc:

"""
from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, Flatten, concatenate, Dense, MaxPooling1D, \
    BatchNormalization

from models.keras_base_model import KerasBaseModel


class MultiTextCNN(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(MultiTextCNN, self).__init__(config, **kwargs)

    def build(self):
        input_text = Input(shape=(self.max_len,))

        embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                    weights=[self.word_embeddings],
                                    trainable=self.config.word_embed_trainable)(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        filter_lengths = [2, 3, 4, 5]
        conv_layers = []
        for filter_length in filter_lengths:
            conv_layer_1 = Conv1D(filters=300, kernel_size=filter_length, strides=1,
                                  padding='valid', activation='relu')(text_embed)
            bn_layer_1 = BatchNormalization()(conv_layer_1)
            conv_layer_2 = Conv1D(filters=300, kernel_size=filter_length, strides=1,
                                  padding='valid', activation='relu')(bn_layer_1)
            bn_layer_2 = BatchNormalization()(conv_layer_2)
            maxpooling = MaxPooling1D(pool_size=self.max_len - 2 * filter_length + 2)(bn_layer_2)
            flatten = Flatten()(maxpooling)
            conv_layers.append(flatten)
        sentence_embed = concatenate(inputs=conv_layers)

        dense_layer = Dense(256, activation='relu')(sentence_embed)
        output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model
