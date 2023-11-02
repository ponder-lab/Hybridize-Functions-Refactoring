# From  https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values

import tensorflow as tf
from nose.tools import assert_raises


class MyClass:

  def __init__(self):
    self.field = None


external_list = []
external_object = MyClass()


@tf.function
def leaky_function():
  a = tf.constant(1)
  external_list.append(a)  # Bad - leaks tensor
  external_object.field = a  # Bad - leaks tensor


assert len(external_list) == 0
assert external_object is not None
assert external_object.field is None
leaky_function()
assert len(external_list) == 1
assert external_object is not None
assert external_object.field is not None

with assert_raises(TypeError):
  assert external_object.field == tf.constant(1)
