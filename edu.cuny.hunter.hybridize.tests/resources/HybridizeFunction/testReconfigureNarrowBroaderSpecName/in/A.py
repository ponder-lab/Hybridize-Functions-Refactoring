# A broader supplied signature whose spec carries a name (#808). The narrowed signature would drop the name, which changes the traced
# placeholders and the function's structured input signature, so the narrowing is declined.
import tensorflow as tf

INPUT_NAME = "t"


@tf.function(
    input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32, name=INPUT_NAME)]
)
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
