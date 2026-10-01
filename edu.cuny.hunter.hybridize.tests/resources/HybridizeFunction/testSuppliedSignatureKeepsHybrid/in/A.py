import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.int32)])
def declared(n):
    # Called with a Python int, which the analysis doesn't see as a tensor; the signature converts it to one.
    return n + 1


@tf.function
def undeclared(n):
    return n + 1


declared(3)
undeclared(3)
