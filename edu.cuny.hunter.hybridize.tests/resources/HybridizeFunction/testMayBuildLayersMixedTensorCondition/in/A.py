import tensorflow as tf


class Gated(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(4)
        self.extra = tf.keras.layers.Dense(4)

    def call(self, x, mode=None):
        x = self.a(x)

        if mode:
            x = self.extra(x)

        return x


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.gated = Gated()

    def call(self, x, mode=None):
        return self.gated(x, mode)


model = Net()
model(tf.ones([2, 4]), None)
model(tf.ones([2, 4]), tf.constant(True))
