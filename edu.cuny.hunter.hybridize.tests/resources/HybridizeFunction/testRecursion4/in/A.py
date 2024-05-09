# From https://www.tensorflow.org/guide/function#recursive_tffunctions_are_not_supported.

import tensorflow as tf
from nose.tools import assert_raises


@tf.function
def recursive_fn(n):
    if n > 0:
        return recursive_fn(n - 1)
    else:
        return 1


with assert_raises(Exception):
    recursive_fn(tf.constant(5))  # Bad - maximum recursion error.
