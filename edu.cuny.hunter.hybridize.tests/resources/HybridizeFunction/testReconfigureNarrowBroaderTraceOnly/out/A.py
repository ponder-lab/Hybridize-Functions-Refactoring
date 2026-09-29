# A broader supplied signature on a function whose concrete function is only traced, not exported (#808). A get_concrete_function call
# that only forces a trace fixes no external interface, so the closed-world narrowing still applies.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
    f.get_concrete_function()
