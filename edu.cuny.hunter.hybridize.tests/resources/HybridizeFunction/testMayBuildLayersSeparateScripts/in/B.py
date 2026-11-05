import tensorflow as tf


class Block(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c_fc = tf.keras.layers.Dense(4, name="c_fc")

    def call(self, x):
        return self.c_fc(x)


class Projection(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.c_proj = tf.keras.layers.Dense(4, name="c_proj")

    def call(self, x):
        return self.c_proj(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.block = Block()
        self.projection = Projection()

    def call(self, x):
        return self.projection(self.block(x))
