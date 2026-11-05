# Fixture for a tensor function whose other parameter is passed only a value the analysis doesn't
# model, the result of string formatting: nothing says whether `scale`'s `name` is a primitive.
# `repeat`'s `n` is passed int literals, and `defaulted`'s `n` always takes its default.
# `absorbing`'s rest slots are passed nothing, and `keyword`'s keyword-only `n` takes its default.
import tensorflow as tf


def scale(x, name):
    return x * 2.0


def repeat(x, n):
    return x * 2.0


def defaulted(x, n=3):
    return x * 2.0


def absorbing(x, *args, **kwargs):
    return x * 2.0


def keyword(x, *, n=3):
    return x * 2.0


print(scale(tf.constant(1.0), "_%d" % 3))
print(repeat(tf.constant(1.0), 1))
print(repeat(tf.constant(2.0), 2))
print(defaulted(tf.constant(1.0)))
print(absorbing(tf.constant(1.0)))
print(keyword(tf.constant(1.0)))
