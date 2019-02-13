# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_cnnrnn_model.py

@time: 2019/2/11 19:19

@desc:

"""


from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, GRU, concatenate, Dense, Bidirectional, \
    MaxPooling1D, GlobalMaxPooling1D

from models.keras_base_model import KerasBaseModel


class CNNRNN(KerasBaseModel):
    def __init__(self, config):
        super(CNNRNN, self).__init__(config)

    def build(self):
        input_text = Input(shape=(self.max_len,))

        embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                    weights=[self.word_embeddings],
                                    trainable=self.config.word_embed_trainable)(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        conv_layer = Conv1D(300, kernel_size=3, padding="valid", activation='relu')(text_embed)
        conv_max_pool = MaxPooling1D(pool_size=2)(conv_layer)

        gru_layer = Bidirectional(GRU(self.config.rnn_units, return_sequences=True))(conv_max_pool)
        sentence_embed = GlobalMaxPooling1D()(gru_layer)

        dense_layer = Dense(256, activation='relu')(sentence_embed)
        if self.config.loss_function == 'binary_crossentropy':
            output = Dense(1, activation='sigmoid')(dense_layer)
        else:
            output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss=self.config.loss_function, metrics=['acc'], optimizer=self.config.optimizer)
        return model
