# Already-hybrid function whose `@tf.function(reduce_retracing=True)` decorator already carries a non-`input_signature` keyword
# argument. Analysis selects `RECONFIGURE`; the source-write appends `, input_signature=[tf.TensorSpec(...)]` at the end of the
# existing argument list, after the `reduce_retracing=True` keyword.
import tensorflow as tf


@tf.function(reduce_retracing=True)
def f(x):
    return x + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
