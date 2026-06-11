# Modify path (#596), incomparable case. The decorator pins a float32 dtype, but the call site passes an int32 tensor, so the
# supplied and inferred dtypes are incomparable. The supplied signature is overwritten with a warning (it changes the inputs accepted
# at runtime). The supplied signature intentionally disagrees with the call site, so this fixture is analyzed statically.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2))
