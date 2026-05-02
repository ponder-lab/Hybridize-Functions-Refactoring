import tensorflow as tf


# Three positional args: `func=None`, `input_signature=None`, `autograph=False`.
# Equivalent to `@tf.function(autograph=False)` but written with positional args.
# Tests #108: positional `autograph` (position 2) should be detected.
@tf.function(None, None, False)
def func(x):
    return x


if __name__ == "__main__":
    number = tf.constant([1.0, 1.0])
    func(number)
