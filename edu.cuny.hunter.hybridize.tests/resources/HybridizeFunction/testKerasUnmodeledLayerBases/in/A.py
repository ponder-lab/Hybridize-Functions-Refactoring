import keras
import tensorflow as tf


# Keras layers whose bases the engine has no summary class for: `keras.layers.Layer`,
# the same class as `tf.keras.layers.Layer` spelled through the standalone package, and
# `MaxPooling2D`, a built-in layer. Keras supplies `training` from `Layer.__call__`, so
# an input signature covering only `x` raises.
class Doubled(keras.layers.Layer):
    def call(self, x, training=False):
        return x * 2.0


class Pooled(tf.keras.layers.MaxPooling2D):
    def call(self, x, training=False):
        return x * 2.0


assert Doubled()(tf.random.normal([1, 3])).shape == (1, 3)
assert Pooled()(tf.random.normal([1, 4, 4, 3])).shape == (1, 4, 4, 3)
