# Already-hybrid function whose decorator passes `func` positionally (`@tf.function(None)`) and carries no `input_signature`.
# Analysis selects `RECONFIGURE`; the source-write appends `, input_signature=[tf.TensorSpec(...)]` at the END of the argument list.
# Front-insertion would place the keyword before the `None` positional argument, producing a Python syntax error (#595 review).
import tensorflow as tf


@tf.function(None, input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(x):
    return x + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
