import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def my_function(x):
    return tf.constant(1)


my_function(tf.constant(2))
