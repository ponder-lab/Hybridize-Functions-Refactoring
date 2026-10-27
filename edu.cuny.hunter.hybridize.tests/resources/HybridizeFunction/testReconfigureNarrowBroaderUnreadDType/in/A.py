# A broader supplied signature whose dtype is given through a name (#808). The supplied-signature parser declines to model a dtype it
# cannot read, so the signature is unmodeled and never narrowed: a dtype the tool did not read is never replaced.
import tensorflow as tf

DTYPE = tf.float32


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=DTYPE)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
