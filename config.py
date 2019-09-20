# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: config.py

@time: 2019/2/8 8:44

@desc:

"""
from keras.optimizers import Adam


class Config(object):
    def __init__(self):
        # input configuration
        self.input_level = 'word'
        self.word_max_len = 128
        self.char_max_len = 200
        self.max_len = {'word': self.word_max_len,
                        'char': self.char_max_len
                        }
        self.han_max_sent = 10
        self.word_embed_dim = 300
        self.word_embed_type = 'glove'
        self.word_embed_trainable = False
        self.word_embeddings = None

        # model structure configuration
        self.exp_name = None
        self.model_name = None
        self.rnn_units = 300
        self.dense_units = 512

        # model training configuration
        self.batch_size = 128
        self.n_epoch = 50
        self.learning_rate = 0.001
        self.optimizer = Adam(self.learning_rate)
        self.dropout = 0.5
        self.l2_reg = 0.001

        # output configuration
        self.n_class = 3

        # checkpoint configuration
        self.checkpoint_dir = 'ckpt'
        self.checkpoint_monitor = 'val_acc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stopping configuration
        self.early_stopping_monitor = 'val_acc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1

