import tensorflow as tf

CONFIG = {"units": 4, "name": "dense"}


class Pair(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(**CONFIG)
        self.second = tf.keras.layers.Dense(**CONFIG)

    def call(self, x):
        return self.second(self.first(x))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.pair = Pair()

    def call(self, x):
        return self.pair(x)


model = Net()
model(tf.ones([2, 4]))
