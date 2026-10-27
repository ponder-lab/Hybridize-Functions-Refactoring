# A broader supplied signature with a comment inside its literal (#808). The narrowed signature would drop the comment, so the narrowing
# is declined.
import tensorflow as tf


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float32),  # Any shape.
    ]
)
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
