# A broader supplied signature in a program that saves an object the analysis cannot resolve (#808). The saved object is an instance of a
# class the summaries do not model, so what it exports is unknown, and the function may be exported through it (here, as an attribute).
# The narrowing is declined.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
    module = tf.Module()
    module.serve = f
    tf.saved_model.save(module, "exported")
