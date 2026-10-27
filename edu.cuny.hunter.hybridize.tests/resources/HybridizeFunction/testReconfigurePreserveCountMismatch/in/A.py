# A supplied signature declaring more parameters than are inferred from the reachable calls (#808). The inferred signature leaves out the
# defaulted parameter no call passes, so the two cannot be compared parameter by parameter, and the signature is left unchanged without
# claiming that a call violates it.
import tensorflow as tf


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
    ]
)
def f(t, training=False):
    return t + 1 if training else t


if __name__ == "__main__":
    f(tf.constant(2.0))
