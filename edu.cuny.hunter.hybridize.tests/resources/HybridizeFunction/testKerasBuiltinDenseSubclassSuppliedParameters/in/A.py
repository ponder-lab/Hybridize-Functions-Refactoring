import tensorflow as tf


# A second built-in base for #1099: `Dense`, whose own ancestry (`Layer`) lives in
# TensorFlow's sources. Keras supplies `training` from `Layer.__call__`, so the
# signature must be withheld as for a direct `Layer` subclass. The test pins the
# signature inferred today, pending wala/ML#1098.
class Affine(tf.keras.layers.Dense):
    def call(self, x, training=False):
        return tf.matmul(x, self.kernel) + self.bias


y = Affine(2)(tf.random.normal([1, 3]))
assert y.shape == (1, 2)
