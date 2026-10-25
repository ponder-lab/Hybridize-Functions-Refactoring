import os
import tempfile

import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):

    def call(self, x, training=False):
        return super().call(x, training)


class Block(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = BatchNormalization()
        self.second = BatchNormalization()

    def call(self, x):
        return self.second(self.first(x))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.block = Block()

    def call(self, x):
        return self.block(x)


model = Net()
model(tf.ones([2, 4]))
model.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
