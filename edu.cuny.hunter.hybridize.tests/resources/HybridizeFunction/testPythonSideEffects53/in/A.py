# From  https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values

import tensorflow as tf


class MyClass:

  def __init__(self):
    self.field = None


# external_list = []
external_object = MyClass()


def not_leaky_function():
  a = tf.constant(1)
  # external_list.append(a)  # Bad - leaks tensor
  return external_object.field


# assert len(external_list) == 0
assert external_object is not None
assert external_object.field is None
not_leaky_function()
# assert len(external_list) == 1
assert external_object is not None
assert external_object.field is None