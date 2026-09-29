# A broader supplied signature whose narrowing would change a rank the function reads statically (#808). The body reads the rank into a
# shape. The supplied signature leaves the rank unknown and the inferred one fixes it, so narrowing would change the program the
# function traces; the narrowing is declined. Under the supplied signature the read is None at trace time, so this fixture is analyzed
# statically rather than executed.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(t):
    return tf.ones(t.shape.rank + 1)


if __name__ == "__main__":
    f(tf.ones([3]))
