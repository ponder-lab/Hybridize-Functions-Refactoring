import tensorflow as tf


class Block(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = tf.keras.layers.Dense(4)

    def project(self, x):
        return self.d(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(4)


# One call site shares `Block.project` between a layer and a model, so its receiver may be either.
for layer in [Block(), Net()]:
    Block.project(layer, tf.ones([2, 4]))
