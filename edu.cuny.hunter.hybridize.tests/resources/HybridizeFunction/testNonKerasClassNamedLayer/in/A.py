import tensorflow as tf


# Classes named like Keras's base but defined by the program (#1107): neither is a Keras
# layer, so nothing supplies `training`. `Layer` is named `Layer`, and `Scaled`'s base's
# last segment is `Layer`, which a match on base names took for Keras's. The parameter
# is left out of each inferred signature, which runs.
class Layer:
    def call(self, x, training=False):
        return tf.nn.relu(x)


class Scaled(Layer):
    def call(self, x, training=False):
        return tf.nn.relu(x) * 2.0


assert Layer().call(tf.random.normal([1, 3])).shape == (1, 3)
assert Scaled().call(tf.random.normal([1, 3])).shape == (1, 3)
