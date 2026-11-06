import tensorflow as tf


class Projector(tf.keras.layers.Layer):

    def call(self, x):
        return tf.keras.layers.Dense(2)(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.projector = Projector()

    def call(self, x):
        return self.projector(x)


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
