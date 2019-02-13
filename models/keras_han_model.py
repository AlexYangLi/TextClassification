# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_han_model.py

@time: 2019/2/8 13:22

@desc:

"""


from keras.models import Model
from keras.layers import Input, Embedding, Dense, Bidirectional, GRU, Masking, TimeDistributed

from models.keras_base_model import KerasBaseModel
from layers.attention import SelfAttention


class HAN(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(HAN, self).__init__(config, **kwargs)

    def build(self):
        input_text = Input(shape=(self.config.han_max_sent, self.max_len))

        sent_encoded = TimeDistributed(self.word_encoder())(input_text)  # word encoder
        sent_vectors = TimeDistributed(SelfAttention(bias=True))(sent_encoded)  # word attention

        doc_encoded = self.sentence_encoder()(sent_vectors)  # sentence encoder
        doc_vector = SelfAttention(bias=True)(doc_encoded)  # sentence attention

        dense_layer = Dense(256, activation='relu')(doc_vector)
        output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model

    def word_encoder(self):
        input_words = Input(shape=(self.max_len,))
        word_vectors = Embedding(input_dim=self.word_embeddings.shape[0], output_dim=self.word_embeddings.shape[1],
                                 weights=[self.word_embeddings], mask_zero=True,
                                 trainable=self.config.word_embed_trainable)(input_words)
        sent_encoded = Bidirectional(GRU(self.config.rnn_units, return_sequences=True))(word_vectors)
        return Model(input_words, sent_encoded)

    def sentence_encoder(self):
        input_sents = Input(shape=(self.config.han_max_sent, self.config.rnn_units * 2))
        sents_masked = Masking()(input_sents)  # support masking
        doc_encoded = Bidirectional(GRU(self.config.rnn_units, return_sequences=True))(sents_masked)
        return Model(input_sents, doc_encoded)
