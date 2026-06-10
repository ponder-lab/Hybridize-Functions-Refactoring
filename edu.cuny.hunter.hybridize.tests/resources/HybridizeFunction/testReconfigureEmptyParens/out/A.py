# Already-hybrid function with an empty-parentheses `@tf.function()` decorator and a tensor parameter, no `input_signature`. Analysis
# selects `RECONFIGURE`; the source-write inserts `input_signature=[tf.TensorSpec(...)]` between the existing parentheses.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(x):
    return x + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
