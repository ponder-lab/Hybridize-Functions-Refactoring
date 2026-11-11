import os
import tempfile

import tensorflow as tf


class Encoder(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(4)

    def build(self, input_shape):
        self.second = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.second(self.first(x))


class Left(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(name="encoder")

    def call(self, x):
        return self.encoder(x)


class Right(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(name="encoder")

    def call(self, x):
        return self.encoder(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.left = Left()
        self.right = Right()

    def call(self, x):
        return self.right(self.left(self.encoder(x)))


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
model.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
