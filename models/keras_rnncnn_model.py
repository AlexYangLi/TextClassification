# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_rnncnn_model.py

@time: 2019/2/8 11:05

@desc:

"""

from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, GRU, concatenate, Dense, Bidirectional, \
    GlobalAveragePooling1D, GlobalMaxPooling1D

from models.keras_base_model import KerasBaseModel


class RNNCNN(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(RNNCNN, self).__init__(config, **kwargs)

    def build(self):
        input_text = Input(shape=(self.max_len,))

        embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                    weights=[self.word_embeddings],
                                    trainable=self.config.word_embed_trainable)(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        gru_layer = Bidirectional(GRU(self.config.rnn_units, return_sequences=True))(text_embed)

        conv_layer = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(gru_layer)

        avg_pool = GlobalAveragePooling1D()(conv_layer)
        max_pool = GlobalMaxPooling1D()(conv_layer)
        sentence_embed = concatenate([avg_pool, max_pool])

        dense_layer = Dense(256, activation='relu')(sentence_embed)
        output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model
