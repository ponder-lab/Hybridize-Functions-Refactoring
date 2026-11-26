# Fixture for #1323 with #787's suffix rule: the one call passing `training` passes its default,
# so the argument could be deleted, but `mask` follows `training` and needs a spec. An input
# signature covers parameters by position, so `training` still can't be left out, and nothing is
# deleted.
import tensorflow as tf


def f(x, training=False, mask=None):
    if training:
        return tf.reduce_sum(x)
    return tf.reduce_sum(x * mask)


f(tf.constant([1.0, 2.0]), training=False, mask=tf.constant([1.0, 0.0]))
