import os
import tempfile

import tensorflow as tf


class Inner(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = tf.keras.layers.Dense(4, name="dense")

    def call(self, x):
        return self.d(x)


class Stack(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = Inner()
        self.second = Inner()

    def call(self, x):
        return self.second(self.first(x))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.stack = Stack()

    def call(self, x):
        return self.stack(x)


class Saver:

    def __init__(self, model):
        self.model = model

    def save(self, filepath):
        self.model.save_weights(filepath)


model = Net()
model(tf.ones([2, 4]))
