import tensorflow as tf


def not_recursive_fn(n):
    if n > 0:
        return abs(n - 1)
    else:
        return 1


not_recursive_fn(tf.constant(5))
