import os
import tempfile

import tensorflow as tf


class Scale(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.w = self.add_weight(name="kernel", shape=(input_shape[-1],))

    def call(self, x):
        return x * self.w


class Mixed(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(4, name="proj")
        self.scale = Scale(name="proj")

    def call(self, x):
        return self.scale(self.dense(x))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.mixed = Mixed()

    def call(self, x):
        return self.mixed(x)


model = Net()
model(tf.ones([2, 4]))
model.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
