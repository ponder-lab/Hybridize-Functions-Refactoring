# Fixture for #1323: the call passes a name bound to the default's value, which the relaxation
# does not compare, since only literals are compared.
import tensorflow as tf

AXIS = -1


def f(x, axis=-1):
    return tf.reduce_sum(x, axis=axis)


f(tf.constant([1.0, 2.0]), axis=AXIS)
