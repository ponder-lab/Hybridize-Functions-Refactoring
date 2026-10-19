import tensorflow as tf


# Reduces TensorFlow2.0-Examples' YOLOV3 `core/common.py` (#1099): the class extends a
# built-in layer rather than `Layer` itself, and `Layer.__call__` still supplies
# `training=False` on every invocation. An input signature covering only `x` raises on
# the first call ("got keyword argument `training` that was not included in
# input_signature"), so the signature must be withheld; the bare decorator runs.
# The test pins the signature inferred today, pending wala/ML#1098.
class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


y = BatchNormalization()(tf.random.normal([1, 4, 4, 3]))
assert y.shape == (1, 4, 4, 3)
