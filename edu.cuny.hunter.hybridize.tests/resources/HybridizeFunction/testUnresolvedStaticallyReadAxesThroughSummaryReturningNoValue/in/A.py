import tensorflow as tf


def clipped(x, n):
    piece = x[:n]
    return tf.range(piece.shape[-1] - 1, -1, -1)


def boxed(x):
    d = dict(v=x)
    return tf.range(d["v"].shape[-1] - 1, -1, -1)


a = tf.ones((2, 4))
b = tf.ones((2, 5))
one = tf.constant(1)

assert clipped(a, one).shape == (4,)
assert clipped(b, one).shape == (5,)
assert boxed(a).shape == (4,)
assert boxed(b).shape == (5,)
