# Fixture for #1323: `f` is wrapped by a decorator, so its callers reach the wrapper, whose
# parameters the definition's do not describe.
import functools

import tensorflow as tf


def logged(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


@logged
def f(x, axis=-1):
    return tf.reduce_sum(x, axis=axis)


f(tf.constant([1.0, 2.0]), axis=-1)
