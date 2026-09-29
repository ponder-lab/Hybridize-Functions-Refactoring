# A broader supplied signature in a program that converts a concrete function the analysis cannot resolve to a TensorFlow Lite model
# (#808). The concrete function belongs to a loaded model, which the summaries do not model, so what the conversion exports is unknown,
# and the narrowing is declined.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
    loaded = tf.saved_model.load("other")
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [loaded.f.get_concrete_function()]
    )
