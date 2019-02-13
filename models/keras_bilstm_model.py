# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_bilstm_model.py

@time: 2019/2/8 11:00

@desc:

"""

from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Dense, LSTM, Bidirectional, Lambda
import keras.backend as K

from models.keras_base_model import KerasBaseModel


class BiLSTM(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(BiLSTM, self).__init__(config, **kwargs)

    def build(self):
        input_text = Input(shape=(self.max_len,))

        embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                    weights=[self.word_embeddings],
                                    trainable=self.config.word_embed_trainable, mask_zero=True)(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        hidden_states = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))(text_embed)
        global_max_pooling = Lambda(lambda x: K.max(x, axis=1))  # GlobalMaxPooling1D didn't support masking
        sentence_embed = global_max_pooling(hidden_states)

        dense_layer = Dense(256, activation='relu')(sentence_embed)
        output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model
