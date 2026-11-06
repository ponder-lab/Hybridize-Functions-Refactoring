import tensorflow as tf


class Block(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = tf.keras.layers.Dense(4)

    def build(self, input_shape):
        self.lazy = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.lazy(self.eager(x))


class Top(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = Block(name="h0")
        self.second = Block(name="h1")

    def call(self, x):
        return self.second(self.first(x))


class LazyBlock(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = tf.keras.layers.Dense(4)
        self.second = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.second(self.first(x))


class LazyTop(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.block = LazyBlock(name="h0")

    def call(self, x):
        return self.block(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.top = Top(name="top")
        self.lazy_top = LazyTop(name="lazy_top")

    def call(self, x):
        return self.lazy_top(self.top(x))


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
