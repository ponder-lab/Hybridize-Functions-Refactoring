import tensorflow as tf


# A program-defined Keras layer, the base `A.Tripled` extends from another module.
class Doubled(tf.keras.layers.Layer):
    def call(self, x, training=False):
        return x * 2.0
