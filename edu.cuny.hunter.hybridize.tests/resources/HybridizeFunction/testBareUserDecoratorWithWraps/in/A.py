import functools

import tensorflow as tf


def logged(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@logged
def scale(x):
    return x * 2.0


def plain(x):
    return x * 2.0


@logged
@tf.function
def hybrid_scale(x):
    return x * 2.0


y = scale(tf.constant([1.0, 2.0]))
z = plain(tf.constant([1.0, 2.0]))
w = hybrid_scale(tf.constant([1.0, 2.0]))
