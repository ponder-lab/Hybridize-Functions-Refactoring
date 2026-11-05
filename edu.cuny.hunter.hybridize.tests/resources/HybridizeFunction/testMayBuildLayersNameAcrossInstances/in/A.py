import os
import tempfile

import tensorflow as tf


class Block(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c_fc = tf.keras.layers.Dense(4, name="c_fc")

    def call(self, x):
        return self.c_fc(x)


class Stack(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [Block() for _ in range(3)]

    def call(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.stack = Stack()

    def call(self, x):
        return self.stack(x)


model = Net()
model(tf.ones([2, 4]))
model.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
