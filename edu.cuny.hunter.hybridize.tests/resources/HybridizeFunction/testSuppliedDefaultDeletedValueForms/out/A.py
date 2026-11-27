# Fixture for #1323: the calls write the defaults differently, but with equal values: another
# quote form, a digit separator, a trailing zero, a hexadecimal integer, and `None`.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(2,), dtype=tf.float32)])
def f(x, name="a", size=1000, rate=0.1, mask=16, extra=None):
    return tf.reduce_sum(x) * rate


f(tf.constant([1.0, 2.0]))
