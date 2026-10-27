# A broader supplied signature whose narrowing would fix the rank of a parameter whose axis the function reads statically (#808). The
# supplied signature leaves the rank unknown, so the read axis is unknown under it and fixed under the inferred one; the narrowing is
# declined. Under the supplied signature the read is None at trace time, so this fixture is analyzed statically rather than executed.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(t):
    return tf.reshape(t, [t.shape[0], 1])


if __name__ == "__main__":
    f(tf.ones([3]))
