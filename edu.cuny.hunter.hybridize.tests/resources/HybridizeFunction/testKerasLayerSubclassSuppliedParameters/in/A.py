import tensorflow as tf


# The direct-`Layer` control for #1099: the base's last segment is `Layer`, which the
# #881 rule recognized before the built-in subclasses were. Keras supplies `training`
# from `Layer.__call__`, so the signature must be withheld.
class Doubled(tf.keras.layers.Layer):
    def call(self, x, training=False):
        return x * 2.0


y = Doubled()(tf.random.normal([1, 3]))
assert y.shape == (1, 3)
