# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: folding.py

@time: 2019/2/9 20:48

@desc:

"""

import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer


class Folding(Layer):
    def __init__(self, **kwargs):
        super(Folding, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Input into Folding Layer must be a 3D tensor!')
        super(Folding, self).build(input_shape)

    def call(self, inputs):
        # split the tensor along dimension 2 into dimension_axis_size/2
        # which will give us 2 tensors.
        # will raise ValueError if K.int_shape(inputs) is odd
        splits = tf.split(inputs, int(K.int_shape(inputs)[-1] / 2), axis=-1)

        # reduce sums of the pair of rows we have split onto
        reduce_sums = [tf.reduce_sum(split, axis=-1) for split in splits]

        # stack them up along the same axis we have reduced
        row_reduced = tf.stack(reduce_sums, axis=-1)
        return row_reduced

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], int(input_shape[2] / 2)
