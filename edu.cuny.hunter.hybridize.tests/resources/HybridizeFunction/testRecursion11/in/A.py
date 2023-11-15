# From https://www.tensorflow.org/guide/function#recursive_tffunctions_are_not_supported.

import tensorflow as tf


def recursive_fn(n):
  if n > 0:
    return recursive_fn(n - 1)
  else:
    return 1


def not_recursive_fn(n):
  if n > 0:
    return abs(n - 1)
  elif n <= 0:
    return 1
  else:
    return recursive_fn(n)


not_recursive_fn(tf.constant(5))  # Bad - maximum recursion error.
