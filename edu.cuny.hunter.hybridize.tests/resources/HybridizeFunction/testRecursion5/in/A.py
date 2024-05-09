# From https://www.tensorflow.org/guide/function#recursive_tffunctions_are_not_supported.

import tensorflow as tf


@tf.function
def not_recursive_fn(n):
    if n > 0:
        return abs(n - 1)
    else:
        return 1


not_recursive_fn(tf.constant(5))  # Bad - maximum recursion error.
