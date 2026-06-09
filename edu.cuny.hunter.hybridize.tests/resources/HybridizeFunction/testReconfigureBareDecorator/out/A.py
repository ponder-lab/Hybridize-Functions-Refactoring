# Already-hybrid function with a bare `@tf.function` (no parentheses) and a single tensor parameter, no `input_signature`. Analysis
# selects `RECONFIGURE`; the source-write appends a parenthesized `input_signature=[tf.TensorSpec(...)]` argument list right after the
# decorator name.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(x):
    return x + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
