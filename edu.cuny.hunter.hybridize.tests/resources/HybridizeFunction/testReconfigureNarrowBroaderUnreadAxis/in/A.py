# A broader supplied signature whose narrowing changes only an axis the function does not read statically (#808). The body reads axis 1,
# which the supplied and inferred signatures both fix, while the narrowing fixes only axis 0, so the traced program is unchanged and
# the narrowing applies.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)])
def f(t):
    return tf.reshape(t, [t.shape[1], -1])


if __name__ == "__main__":
    f(tf.ones([2, 3]))
