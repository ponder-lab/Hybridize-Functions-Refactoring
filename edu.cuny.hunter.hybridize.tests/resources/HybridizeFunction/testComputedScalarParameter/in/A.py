# Python scalars of unknown value as arguments (wala/ML#1148). Each is a computed `int`, so each distinct value is a new trace under
# `tf.function`.
# - `scale`: `n` is only ever `len(xs)`.
# - `scale_first`: `ns` is a list whose element is `len(xs)`.
# - `wrapped`: `x` is a tensor or a variable built from `len(xs)`, which is no primitive.
# - `summed`: `v` is a list of tensors at one call and `len(xs)` at the other.
import tensorflow as tf


def scale(x, n):
    return x * n


def scale_first(x, ns):
    return x * ns[0]


def wrapped(x):
    return x * 2


def summed(v):
    return tf.reduce_sum(v)


t = tf.ones([2])
xs = [1, 2, 3]

scale(t, len(xs))
scale_first(t, [len(xs)])
wrapped(tf.constant(len(xs)))
wrapped(tf.Variable(len(xs)))
wrapped(tf.fill([2], len(xs)))
wrapped(tf.ones([len(xs)]))
wrapped(tf.convert_to_tensor(len(xs)))
wrapped(tf.range(len(xs)))
summed([t, t])
summed(len(xs))
