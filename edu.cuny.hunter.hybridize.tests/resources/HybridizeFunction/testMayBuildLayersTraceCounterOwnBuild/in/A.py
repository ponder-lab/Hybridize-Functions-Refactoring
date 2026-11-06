import tensorflow as tf


class Mixed(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = tf.keras.layers.Dense(4)

    def build(self, input_shape):
        self.lazy = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.lazy(self.eager(x))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.mixed = Mixed(name="mixed")

    def call(self, x):
        return self.mixed(x)


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
