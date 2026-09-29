# Adjudication path (#596, #808), incomparable case. The decorator pins a float32 dtype, but the call site passes an int32 tensor, so
# the supplied and inferred dtypes are incomparable. The call raises at runtime, so the supplied signature is left unchanged and the
# disagreement is reported. The supplied signature intentionally disagrees with the call site, so this fixture is analyzed statically.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2))
