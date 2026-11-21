import tensorflow as tf


def clipped(x, n):
    piece = x[:n]
    return tf.range(piece.shape[-1] - 1, -1, -1)


def boxed(x):
    d = dict(v=x)
    return tf.range(d["v"].shape[-1] - 1, -1, -1)


def looped(y, x):
    def whole(k):
        return y

    n = tf.cast(x[0, 0], tf.int32)
    piece = None
    for make in (slice, whole):
        piece = make(n)
    return tf.range(piece.shape[-1] - 1, -1, -1)


a = tf.ones((2, 4))
b = tf.ones((2, 5))
one = tf.constant(1)

assert clipped(a, one).shape == (4,)
assert clipped(b, one).shape == (5,)
assert boxed(a).shape == (4,)
assert boxed(b).shape == (5,)
assert looped(tf.ones((3, 4)), a).shape == (4,)
assert looped(tf.ones((3, 5)), a).shape == (5,)
