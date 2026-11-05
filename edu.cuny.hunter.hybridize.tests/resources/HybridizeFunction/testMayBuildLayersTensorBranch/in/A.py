import tensorflow as tf


class Gated(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(4)
        self.b = tf.keras.layers.Dense(4)

    def call(self, x):
        x = self.a(x)

        if tf.reduce_sum(x) > 0:
            x = self.b(x)

        return x


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.gated = Gated()

    def call(self, x):
        return self.gated(x)


model = Net()
model(tf.ones([2, 4]))
model(-tf.ones([3, 4]))
