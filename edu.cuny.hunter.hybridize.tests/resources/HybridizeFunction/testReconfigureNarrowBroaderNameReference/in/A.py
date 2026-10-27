# A broader supplied signature given by name (#808). The module-level constant may be shared with other decorators, so rewriting it for
# this function alone would split it; the narrowing is declined and the signature is left unchanged.
import tensorflow as tf

f_signature = [tf.TensorSpec(shape=None, dtype=tf.float32)]


@tf.function(input_signature=f_signature)
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
