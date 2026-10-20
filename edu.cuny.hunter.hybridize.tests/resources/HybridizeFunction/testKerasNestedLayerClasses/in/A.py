import tensorflow as tf


# Keras layers defined in a function and in a class. Keras supplies `training` from
# `Layer.__call__`, so an input signature covering only `x` raises.
def make_layer():
    class Local(tf.keras.layers.Layer):
        def call(self, x, training=False):
            return x * 2.0

    return Local()


class Outer:
    class Inner(tf.keras.layers.Layer):
        def call(self, x, training=False):
            return x * 3.0


assert make_layer()(tf.random.normal([1, 3])).shape == (1, 3)
assert Outer.Inner()(tf.random.normal([1, 3])).shape == (1, 3)
