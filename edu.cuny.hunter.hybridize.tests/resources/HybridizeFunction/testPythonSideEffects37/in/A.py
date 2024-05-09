# From https://www.tensorflow.org/guide/function#changing_python_global_and_free_variables.
import tensorflow as tf

external_list = [1, 3, 5]


@tf.function
def no_side_effect(x):
    tf.print("No python side effect")
    return external_list[x]


assert len(external_list) == 3
no_side_effect(1)
no_side_effect(1)
no_side_effect(1)
# The list append only happened once!
assert len(external_list) == 3
