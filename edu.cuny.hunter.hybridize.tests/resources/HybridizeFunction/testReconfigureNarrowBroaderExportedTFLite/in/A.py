# A broader supplied signature on a function the program converts to a TensorFlow Lite model (#808). The concrete function is handed to
# the converter, so the function's signature is part of the exported interface and the narrowing is declined.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [f.get_concrete_function()]
    )
