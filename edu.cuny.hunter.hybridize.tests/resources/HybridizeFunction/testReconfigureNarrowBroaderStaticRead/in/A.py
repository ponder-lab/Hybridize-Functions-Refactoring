# A broader supplied signature whose narrowing would change a shape the function reads statically (#808). The body passes axis 0 into a
# reshape target. The supplied signature leaves that axis unknown and the inferred one fixes it, so narrowing would change the program
# the function traces; the narrowing is declined. Under the supplied signature the read is None at trace time, so this fixture is
# analyzed statically rather than executed.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32)])
def f(t):
    return tf.reshape(t, [t.shape[0], 1])


if __name__ == "__main__":
    f(tf.ones([3]))
