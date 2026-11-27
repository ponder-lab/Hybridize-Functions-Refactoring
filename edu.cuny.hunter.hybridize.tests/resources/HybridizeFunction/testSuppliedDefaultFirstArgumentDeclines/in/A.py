# Fixture for #1323: the only call passing `axis` passes its default, but as the call's first
# written argument, with no comma before it to remove along with it, so the argument is not
# deleted.
import tensorflow as tf


def f(x, axis=-1):
    return tf.reduce_sum(x, axis=axis)


f(axis=-1, x=tf.constant([1.0, 2.0]))
