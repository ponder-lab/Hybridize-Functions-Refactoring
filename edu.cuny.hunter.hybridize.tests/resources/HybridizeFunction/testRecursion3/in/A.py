# From https://www.tensorflow.org/guide/function#recursive_tffunctions_are_not_supported

import tensorflow as tf


def recursive_fn2(n):
  if n > 0:
    return recursive_fn(n - 1)
  else:
    return 1


# @tf.function
def recursive_fn(n):
    return recursive_fn2(n)


recursive_fn(tf.constant(5))  # Bad - maximum recursion error.
