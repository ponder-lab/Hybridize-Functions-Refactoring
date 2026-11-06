import tensorflow as tf

CONFIG = {"units": 4, "name": "dense"}
NAMED = {"name": "named"}


class Pair(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(**CONFIG)
        self.second = tf.keras.layers.Dense(**CONFIG)

    def call(self, x):
        return self.second(self.first(x))


class Named(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.d(x)


class Twins(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = Named(**NAMED)
        self.second = Named(**NAMED)

    def call(self, x):
        return self.second(self.first(x))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.pair = Pair()
        self.twins = Twins()

    def call(self, x):
        return self.twins(self.pair(x))


model = Net()
model(tf.ones([2, 4]))
