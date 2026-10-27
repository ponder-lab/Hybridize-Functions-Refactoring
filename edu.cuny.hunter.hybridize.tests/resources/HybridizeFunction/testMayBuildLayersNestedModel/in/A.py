import os
import tempfile

import tensorflow as tf


class Inner(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = tf.keras.layers.Dense(4, name="dense")

    def call(self, x):
        return self.d(x)


class Outer(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.first = Inner()
        self.second = Inner()

    def call(self, x):
        return self.second(self.first(x))


model = Outer()
model(tf.ones([2, 4]))
model.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
