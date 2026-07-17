# Fixture for #787: the case the precondition unblocks. `training` is non-tensor and defaulted,
# and no call site supplies it, so it is omittable and the signature covers `x` alone.
# TensorFlow accepts that: it requires a TensorSpec per *required* argument, not per argument
# (verified on TF 2.9.3), so `@tf.function(input_signature=[tf.TensorSpec(shape=[2],
# dtype=tf.float32)])` on `def f(x, training=True)` is legal. Before #787, `training` was
# reported NON_TENSOR_PARAMETER and dropped the whole signature.
import tensorflow as tf


def f(x, training=True):
    if training:
        return tf.reduce_sum(x)
    return tf.reduce_sum(x) * 2


f(tf.constant([1.0, 2.0]))
