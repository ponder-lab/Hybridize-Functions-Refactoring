# From https://www.tensorflow.org/guide/function#changing_python_global_and_free_variables.
import tensorflow as tf

external_list = []


@tf.function
def side_effect(x):
  tf.print('Python side effect')
  external_list.append(x)


side_effect(1)
side_effect(1)
side_effect(1)
# The list append only happened once!
assert len(external_list) == 1
