import numpy as np
import tensorflow as tf


class Embedded(tf.keras.layers.Layer):

    def __init__(self, weights=None, **kwargs):
        super().__init__(**kwargs)
        if weights is not None:
            self.embedding = tf.keras.layers.Embedding(
                10, 4, weights=[np.array(weights)]
            )
        else:
            self.embedding = tf.keras.layers.Embedding(10, 4)

    def call(self, inputs):
        return self.embedding(inputs)


class Swapped(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(10, 4)

    def swap(self):
        self.embedding = tf.keras.layers.Embedding(10, 4)

    def call(self, inputs):
        return self.embedding(inputs)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.plain = Embedded()
        self.preset = Embedded(np.ones([10, 4]))
        self.swapped = Swapped()

    def call(self, inputs):
        return self.plain(inputs) + self.preset(inputs) + self.swapped(inputs)


model = Net()
model(tf.constant([[1, 2]]))
model.swapped.swap()
model(tf.constant([[1, 2, 3]]))
