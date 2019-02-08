# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: base_model.py

@time: 2019/2/1 14:03

@desc:

"""

import abc


class BaseModel(object):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.model = None

    @abc.abstractmethod
    def build(self):
        """Build the model"""

    @abc.abstractmethod
    def train(self, data_train, data_dev):
        """Train the model"""

    @abc.abstractmethod
    def load_weights(self, filename):
        """Load weights from the `filename`"""

    @abc.abstractmethod
    def evaluate(self, data):
        """Evaluate the model on the provided data"""

    @abc.abstractmethod
    def predict(self, data):
        """Predict for the provided data"""
