import tensorflow as tf


# Positional `func=None` (position 0) followed by positional `input_signature` (position 1).
# Equivalent to `@tf.function(input_signature=(...,))` but written with positional args.
# Tests #108: positional args to @tf.function should be detected.
@tf.function(None, (tf.TensorSpec(shape=[None], dtype=tf.float32),))
def func(x):
    return x


if __name__ == "__main__":
    number = tf.constant([1.0, 1.0])
    func(number)
