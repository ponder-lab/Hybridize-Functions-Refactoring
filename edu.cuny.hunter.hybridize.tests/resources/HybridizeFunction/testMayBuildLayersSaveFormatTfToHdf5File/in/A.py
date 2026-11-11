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


stack = Stack()
stack(tf.ones([2, 4]))
model = tf.keras.Sequential([stack])
model(tf.ones([2, 4]))
model.save(os.path.join(tempfile.mkdtemp(), "m.h5"), save_format="tf")
