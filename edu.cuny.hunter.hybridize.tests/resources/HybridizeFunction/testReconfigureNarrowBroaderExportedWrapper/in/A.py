# A broader supplied signature on a function the program exports as a SavedModel signature through another tf.function wrapping it
# (#808). The wrapper holds the function, so the narrowing is declined.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(t):
    return t + 1


class N(tf.Module):
    pass


if __name__ == "__main__":
    f(tf.constant(2.0))
    tf.saved_model.save(N(), "exported", signatures=tf.function(f))
