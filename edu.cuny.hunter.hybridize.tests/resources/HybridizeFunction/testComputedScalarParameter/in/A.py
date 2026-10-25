# Python scalars of unknown value as arguments (wala/ML#1148). Each is a computed `int`, so each distinct value is a new trace under
# `tf.function`.
# - `scale`: `n` is only ever `len(xs)`.
# - `scale_first`: `ns` is a list whose element is `len(xs)`.
# - `wrapped`: `x` is a tensor or a variable built from `len(xs)`, which is no primitive.
# - `summed`: `v` is a list of tensors at one call and `len(xs)` at the other.
# - `boxed`: `box` is an object of a program class whose field is `len(xs)`. `tf.function` keys it by its identity, and a trace reads
#   the field at trace time, so it's a primitive, as `Box(3)` would be.
# - `layered`: `layer` is a Keras layer whose units are `len(xs)`, which should be no primitive.
import tensorflow as tf


def scale(x, n):
    return x * n


def scale_first(x, ns):
    return x * ns[0]


def wrapped(x):
    return x * 2


def summed(v):
    return tf.reduce_sum(v)


class Box:
    def __init__(self, n):
        self.n = n


def boxed(x, box):
    return x * box.n


def layered(x, layer):
    return layer(x)


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
boxed(t, Box(len(xs)))
layered(tf.ones([1, 2]), tf.keras.layers.Dense(len(xs)))
