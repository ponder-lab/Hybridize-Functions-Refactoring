# Fixture for #1323: `p` is keyword-only, the second of them, so a call not naming it omits it,
# even though its second written argument, in the slot `p`'s index would name, equals `p`'s
# default. Only `p=-1` is deleted. `b` is a required number, which blocks the signature itself.
import tensorflow as tf


def f(a, b, *, q=0, p=-1):
    return tf.reduce_sum(a) * b * p + q


f(tf.constant([1.0, 2.0]), -1)
f(tf.constant([3.0, 4.0]), -1, p=-1)
