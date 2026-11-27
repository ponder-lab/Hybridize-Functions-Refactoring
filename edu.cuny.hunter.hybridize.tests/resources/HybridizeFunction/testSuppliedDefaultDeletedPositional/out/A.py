# Fixture for #1323: the default is passed as the call's last positional argument.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(2,), dtype=tf.float32)])
def f(x, axis=-1):
    return tf.reduce_sum(x, axis=axis)


f(tf.constant([1.0, 2.0]))
f(tf.constant([3.0, 4.0]))
