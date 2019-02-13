# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_dcnn_model.py

@time: 2019/2/8 13:03

@desc:

"""

from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Conv1D, Flatten, Dense, ZeroPadding1D, ReLU

from models.keras_base_model import KerasBaseModel
from layers.kmaxpooling import KMaxPooling
from layers.folding import Folding


class DCNN(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(DCNN, self).__init__(config, **kwargs)

    def build(self):
        input_text = Input(shape=(self.max_len,))

        embedding_layer = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                    weights=[self.word_embeddings],
                                    trainable=self.config.word_embed_trainable)(input_text)
        text_embed = SpatialDropout1D(0.2)(embedding_layer)

        # wide convolution
        zero_padded_1 = ZeroPadding1D((6, 6))(text_embed)
        conv_1 = Conv1D(filters=128, kernel_size=7, strides=1, padding='valid')(zero_padded_1)
        # dynamic k-max pooling
        k_maxpool_1 = KMaxPooling(int(self.max_len / 3 * 2))(conv_1)
        # non-linear feature function
        non_linear_1 = ReLU()(k_maxpool_1)

        # wide convolution
        zero_padded_2 = ZeroPadding1D((4, 4))(non_linear_1)
        conv_2 = Conv1D(filters=128, kernel_size=5, strides=1, padding='valid')(zero_padded_2)
        # dynamic k-max pooling
        k_maxpool_2 = KMaxPooling(int(self.max_len / 3 * 1))(conv_2)
        # non-linear feature function
        non_linear_2 = ReLU()(k_maxpool_2)

        # wide convolution
        zero_padded_3 = ZeroPadding1D((2, 2))(non_linear_2)
        conv_3 = Conv1D(filters=128, kernel_size=5, strides=1, padding='valid')(zero_padded_3)
        # folding
        folded = Folding()(conv_3)
        # dynamic k-max pooling
        k_maxpool_3 = KMaxPooling(k=10)(folded)
        # non-linear feature function
        non_linear_3 = ReLU()(k_maxpool_3)

        sentence_embed = Flatten()(non_linear_3)

        dense_layer = Dense(256, activation='relu')(sentence_embed)
        output = Dense(self.n_class, activation='softmax')(dense_layer)

        model = Model(input_text, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model
