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


def load_model(path):
    return os.path.exists(path)


def restore(directory):
    return load_model(os.path.join(directory, "m.h5"))


model = Net()
model(tf.ones([2, 4]))
restore(tempfile.mkdtemp())
