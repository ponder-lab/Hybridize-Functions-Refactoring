# Already-hybrid function that already supplies an `input_signature`. Per the presence/parse three-state contract, a supplied
# signature must not be clobbered, so `RECONFIGURE` is NOT selected even with inference enabled (validate-then-overwrite is future
# work). This pins the deferral of the supplied-signature case.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(x):
    return x + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
