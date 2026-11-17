import os
import tempfile

import tensorflow as tf


def helper_a(layer, x):
    return layer(x)


def helper_b(layer, x):
    return layer(x)


class Inner(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3, name="dense")

    def call(self, x):
        return helper_a(self.dense, x)


class Block(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.inner = Inner()
        self.dense = tf.keras.layers.Dense(3, name="dense")

    def call(self, x):
        return helper_b(self.dense, self.inner(x))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.block = Block()

    def call(self, x):
        return self.block(x)


net = Net()
net(tf.ones([2, 3]))
net.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
