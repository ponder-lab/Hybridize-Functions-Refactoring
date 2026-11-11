import os
import tempfile

import tensorflow as tf


class Block(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(4, name="dense")

    def call(self, x):
        return self.dense(x)


class Indexed(tf.keras.layers.Layer):

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        self.blocks = []
        for i in range(self.n):
            self.blocks.append(Block(name="h{}".format(i)))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.first = Indexed(2, name="first")
        self.second = Indexed(2, name="second")

    def call(self, x):
        return self.second(self.first(x))


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
model.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
