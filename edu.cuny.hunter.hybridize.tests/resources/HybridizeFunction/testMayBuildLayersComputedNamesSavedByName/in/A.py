import os
import tempfile

import tensorflow as tf


class Stack(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hs = [tf.keras.layers.Dense(4, name=f"h{i}") for i in range(3)]

    def call(self, x):
        return self.hs[2](self.hs[1](self.hs[0](x)))


class Formatted(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hs = [tf.keras.layers.Dense(4, name="g{}".format(i)) for i in range(3)]

    def call(self, x):
        return self.hs[2](self.hs[1](self.hs[0](x)))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.stack = Stack()
        self.formatted = Formatted()

    def call(self, x):
        return self.formatted(self.stack(x))


model = Net()
model(tf.ones([2, 4]))
model.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
