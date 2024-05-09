# From  https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values

import tensorflow as tf

x = None


@tf.function
def leaky_function(a):
    global x
    x = a + 1  # Bad - leaks local tensor
    # return a + 2


correct_a = leaky_function(tf.constant(1))

# print(correct_a.numpy())  # Good - value obtained from function's returns
try:
    x.numpy()  # Bad - tensor leaked from inside the function, cannot be used here
except AttributeError as expected:
    print(expected)
