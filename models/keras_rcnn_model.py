# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_rcnn_model.py

@time: 2019/2/8 12:57

@desc:

"""

from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, Flatten, Dense, Lambda, LSTM, concatenate
import keras.backend as K

from models.keras_base_model import KerasBaseModel


class RCNN(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(RCNN, self).__init__(config, **kwargs)

    def build(self):
        input_text = Input(shape=(self.max_len,))

        embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                    weights=[self.word_embeddings],
                                    trainable=self.config.word_embed_trainable)(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        # We shift the document to the right to obtain the left-side contexts
        l_embedding = Lambda(lambda x: K.concatenate([K.zeros(shape=(K.shape(x)[0], 1, K.shape(x)[-1])),
                                                      x[:, :-1]], axis=1))(text_embed)
        # We shift the document to the left to obtain the right-side contexts
        r_embedding = Lambda(lambda x: K.concatenate([K.zeros(shape=(K.shape(x)[0], 1, K.shape(x)[-1])),
                                                      x[:, 1:]], axis=1))(text_embed)
        # use LSTM RNNs instead of vanilla RNNs as described in the paper.
        forward = LSTM(self.config.rnn_units, return_sequences=True)(l_embedding)  # See equation (1)
        backward = LSTM(self.config.rnn_units, return_sequences=True, go_backwards=True)(r_embedding)  # See equation (2)
        # Keras returns the output sequences in reverse order.
        backward = Lambda(lambda x: K.reverse(x, axes=1))(backward)
        together = concatenate([forward, text_embed, backward], axis=2)  # See equation (3).

        # use conv1D instead of TimeDistributed and Dense
        semantic = Conv1D(self.config.rnn_units, kernel_size=1, activation="tanh")(together)  # See equation (4).
        sentence_embed = Lambda(lambda x: K.max(x, axis=1))(semantic)  # See equation (5).

        dense_layer = Dense(256, activation='relu')(sentence_embed)
        output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model
