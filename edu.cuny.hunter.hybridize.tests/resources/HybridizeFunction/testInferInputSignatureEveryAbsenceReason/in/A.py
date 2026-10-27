import tensorflow as tf


# Receives tensors of two dtypes and `None`, so two reasons hold for `x` at once.
def none_and_mixed(x):
    if x is None:
        return tf.constant(0.0)
    return tf.reduce_sum(x)


# Receives tensors of two dtypes only, so one reason holds.
def mixed(x):
    return tf.reduce_sum(x)


none_and_mixed(tf.constant([1, 2]))
none_and_mixed(tf.constant([1.0, 2.0]))
none_and_mixed(None)

mixed(tf.constant([1, 2]))
mixed(tf.constant([1.0, 2.0]))
