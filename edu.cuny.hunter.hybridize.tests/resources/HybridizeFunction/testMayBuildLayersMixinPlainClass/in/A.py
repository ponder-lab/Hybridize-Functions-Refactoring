import tensorflow as tf


class ScaleMixin:

    def scale(self, x):
        return x * 2.0


class Projector(ScaleMixin):

    def __init__(self):
        self.d = tf.keras.layers.Dense(4)

    def project(self, x):
        return self.d(self.scale(x))


Projector().project(tf.ones([2, 4]))
