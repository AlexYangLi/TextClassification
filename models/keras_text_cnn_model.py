# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_text_cnn_model.py

@time: 2019/2/8 10:35

@desc:

"""

from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, MaxPool1D, Flatten, concatenate, Dense

from models.keras_base_model import KerasBaseModel


class TextCNN(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(TextCNN, self).__init__(config, **kwargs)

    def build(self):
        input_text = Input(shape=(self.max_len,))

        embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                    weights=[self.word_embeddings],
                                    trainable=self.config.word_embed_trainable)(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        filter_lengths = [2, 3, 4, 5]
        conv_layers = []
        for filter_length in filter_lengths:
            conv_layer = Conv1D(filters=300, kernel_size=filter_length, padding='valid',
                                strides=1, activation='relu')(text_embed)
            maxpooling = MaxPool1D(pool_size=self.max_len - filter_length + 1)(conv_layer)
            flatten = Flatten()(maxpooling)
            conv_layers.append(flatten)
        sentence_embed = concatenate(inputs=conv_layers)

        dense_layer = Dense(256, activation='relu')(sentence_embed)
        output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model
