import tensorflow as tf


class Pair(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(4, name="dense")
        self.second = tf.keras.layers.Dense(4, name="dense")

    def call(self, x):
        return self.second(self.first(x))


class Mlp(tf.keras.layers.Layer):

    def __init__(self, name="mlp", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.dense(x)


class Twins(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = Mlp()
        self.second = Mlp()

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
