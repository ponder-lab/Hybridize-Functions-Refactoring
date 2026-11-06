import tensorflow as tf


class Pair(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(4)
        self.second = tf.keras.layers.Dense(4, name="dense")

    def call(self, x):
        return self.second(self.first(x))


class Normalized(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.LayerNormalization()
        self.second = tf.keras.layers.LayerNormalization(name="layer_normalization")

    def call(self, x):
        return self.second(self.first(x))


class Apart(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(4)
        self.second = tf.keras.layers.Dense(4, name="proj")

    def call(self, x):
        return self.second(self.first(x))


class Aliased(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Convolution2D(4, 1)
        self.second = tf.keras.layers.Conv2D(4, 1, name="conv2d")

    def call(self, x):
        return tf.reshape(
            self.second(self.first(tf.reshape(x, [-1, 2, 2, 1]))), [-1, 16]
        )


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.pair = Pair(name="pair")
        self.normalized = Normalized(name="normalized")
        self.apart = Apart(name="apart")
        self.aliased = Aliased(name="aliased")

    def call(self, x):
        return self.apart(self.normalized(self.pair(x))), self.aliased(x)


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
