import tensorflow as tf


class Unit(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.d(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.first = Unit()
        self.second = Unit()

    def call(self, x):
        return self.second(self.first(x))


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
