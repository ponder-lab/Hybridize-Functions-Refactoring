# From https://www.tensorflow.org/guide/function#recursive_tffunctions_are_not_supported.

import tensorflow as tf


def recursive_fn(n):
  if n > 0:
    tf.print('tracing')
    return recursive_fn(n - 1)
  else:
    return 1


recursive_fn(5)  # Warning - multiple tracings
