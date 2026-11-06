import tensorflow as tf


class Fallback(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fast = tf.keras.layers.Dense(4)
        self.slow = tf.keras.layers.Dense(4)

    def call(self, x):
        try:
            return self.fast(x)
        except ValueError:
            return self.slow(x)


class Broad(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fast = tf.keras.layers.Dense(4)
        self.slow = tf.keras.layers.Dense(4)

    def call(self, x):
        try:
            return self.fast(x)
        except Exception:
            return self.slow(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.fallback = Fallback()
        self.broad = Broad()

    def call(self, x):
        return self.fallback(x) + self.broad(x)


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([2, 5]))
