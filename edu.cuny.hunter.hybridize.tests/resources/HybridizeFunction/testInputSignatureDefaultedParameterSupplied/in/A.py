# Fixture for #787: `training` is non-tensor and defaulted, but the second call site passes it
# explicitly, so it cannot be omitted. Omitting it would cut the hybridized function's arity to
# one, and `f(tf.constant([1.0, 2.0]), False)` would then raise
# `TypeError: f(x, training) specifies 1 positional arguments, but got 2` (verified on TF 2.9.3).
# The parameter must therefore keep blocking, as DEFAULTED_PARAMETER_SUPPLIED rather than
# NON_TENSOR_PARAMETER: the remedy is to stop passing it, not to make it a tensor.
import tensorflow as tf


def f(x, training=True):
    if training:
        return tf.reduce_sum(x)
    return tf.reduce_sum(x) * 2


f(tf.constant([1.0, 2.0]))
f(tf.constant([3.0, 4.0]), False)
