# Eager function with a single tensor parameter. Analysis selects `CONVERT_TO_HYBRID` and `inferInputSignature` produces
# `[tf.TensorSpec(shape=(), dtype=tf.float32)]`. The test flips `inferInputSignatures` on and asserts the source-write emits
# the `input_signature=...` argument into the generated `@tf.function(...)` decorator.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(x):
    return x + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
