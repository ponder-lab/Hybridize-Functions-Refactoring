import tensorflow as tf

from layers import Doubled


# A Keras layer whose base is a Keras layer defined in another module. Keras supplies
# `training` from `Layer.__call__`, so an input signature covering only `x` raises.
class Tripled(Doubled):
    def call(self, x, training=False):
        return x * 3.0


y = Tripled()(tf.random.normal([1, 3]))
assert y.shape == (1, 3)
