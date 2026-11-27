# Fixture for #1323: the default is a raw string, and the one call passing it passes the same raw
# string, so the argument, prefix included, is deleted.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(2,), dtype=tf.float32)])
def f(x, pattern=r"\d+"):
    tf.print(pattern)
    return tf.reduce_sum(x)


f(tf.constant([1.0, 2.0]))
f(tf.constant([3.0, 4.0]))
