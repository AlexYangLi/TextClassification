# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: kmaxpooling.py

@time: 2019/2/8 13:08

@desc:

"""

import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.contrib.framework import sort


class KMaxPooling(Layer):
    """
    Implemetation of temporal k-max pooling layer, which was first proposed in Kalchbrenner et al.
    [http://www.aclweb.org/anthology/P14-1062]
    "A Convolutional Neural Network for Modelling Sentences"
    This layer allows to detect the k most important features in a sentence, independent of their
    specific position, preserving their relative order.
    """
    def __init__(self, k=1, **kwargs):
        self.k = k

        super(KMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Input into KMaxPooling muse be a 3D tensor!')
        if self.k > input_shape[1]:
            raise ValueError('detect `%d` most important features from `%d` timesteps is not allowed' %
                             (self.k, input_shape[1]))
        super(KMaxPooling, self).build(input_shape)

    def call(self, inputs):
        """
        Reference: https://stackoverflow.com/questions/51299181/how-to-implement-k-max-pooling-in-tensorflow-or-keras
        The key point is preserving the relative order
        """
        permute_inputs = K.permute_dimensions(inputs, (0, 2, 1))
        flat_permute_inputs = tf.reshape(permute_inputs, (-1,))
        topk_indices = sort(tf.nn.top_k(permute_inputs, k=self.k)[1])

        all_indices = tf.reshape(tf.range(K.shape(flat_permute_inputs)[0]), K.shape(permute_inputs))
        to_sum_indices = tf.expand_dims(tf.gather(all_indices, 0, axis=-1), axis=-1)
        topk_indices += to_sum_indices

        flat_topk_indices = tf.reshape(topk_indices, (-1, ))
        topk_output = tf.reshape(tf.gather(flat_permute_inputs, flat_topk_indices), K.shape(topk_indices))

        return K.permute_dimensions(topk_output, (0, 2, 1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.k, input_shape[-1]