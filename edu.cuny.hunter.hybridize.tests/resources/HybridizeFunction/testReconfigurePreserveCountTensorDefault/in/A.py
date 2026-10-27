# A supplied signature declaring fewer parameters than are inferred, where the extra parameter has a tensor default no call passes (#808).
# TensorFlow fills in the default, so no call is shown to violate the signature, and it is left unchanged with a warning.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t, scale=tf.constant(2.0)):
    return t * scale


if __name__ == "__main__":
    f(tf.constant(1.0))
