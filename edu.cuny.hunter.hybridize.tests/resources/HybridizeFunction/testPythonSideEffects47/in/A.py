# From  https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values

import tensorflow as tf
from nose.tools import assert_raises


@tf.function
def leaky_function(a):
    global x
    x = a + 1  # Bad - leaks local tensor
    return x  # Good - uses local tensor


correct_a = leaky_function(tf.constant(1))

print(correct_a.numpy())  # Good - value obtained from function's returns
try:
    x.numpy()  # Bad - tensor leaked from inside the function, cannot be used here
except AttributeError as expected:
    print(expected)


@tf.function
def captures_leaked_tensor(b):
    b += x  # Bad - `x` is leaked from `leaky_function`
    return b


with assert_raises(TypeError):
    captures_leaked_tensor(tf.constant(2))
