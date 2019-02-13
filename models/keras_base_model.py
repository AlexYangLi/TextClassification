# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_base_model.py

@time: 2019/2/3 17:14

@desc:

"""

import os
import abc
import logging

from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.base_model import BaseModel
from utils.metrics import eval_acc


class KerasBaseModel(BaseModel):
    def __init__(self, config, **kwargs):
        super(KerasBaseModel, self).__init__()
        self.config = config
        self.level = self.config.input_level
        self.max_len = self.config.max_len[self.config.input_level]
        self.word_embeddings = config.word_embeddings
        self.n_class = config.n_class

        self.callbacks = []
        self.init_callbacks()

        self.model = self.build(**kwargs)

    def init_callbacks(self):
        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name),
            monitor=self.config.checkpoint_monitor,
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_save_weights_mode,
            verbose=self.config.checkpoint_verbose
        ))

        self.callbacks.append(EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            verbose=self.config.early_stopping_verbose
        ))

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def load_best_model(self):
        logging.info('loading model checkpoint: %s.hdf5\n' % self.config.exp_name)
        self.load_weights(os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name))
        logging.info('Model loaded')

    @abc.abstractmethod
    def build(self):
        """Build the model"""

    def train(self, data_train, data_dev=None):
        x_train, y_train = data_train

        logging.info('start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epoch,
                       validation_split=0.1, validation_data=data_dev, callbacks=self.callbacks)
        logging.info('training end...')

    def evaluate(self, data):
        input_data, label = data
        prediction = self.predict(input_data)
        acc = eval_acc(label, prediction)
        logging.info('acc : %f', acc)
        return acc

    def predict(self, data):
        return self.model.predict(data)
