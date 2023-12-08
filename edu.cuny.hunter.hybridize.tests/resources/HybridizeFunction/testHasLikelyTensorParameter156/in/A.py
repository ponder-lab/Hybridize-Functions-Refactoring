# From https://www.tensorflow.org/guide/function#usage.

import tensorflow as tf


@tf.function(experimental_follow_type_hints=True)
def add(t: tuple[tf.Tensor, tf.Tensor]):
  return t[0] + t[1]


arg = (2, 2)
assert type(arg) == tuple
result = add(arg)
print(result)
