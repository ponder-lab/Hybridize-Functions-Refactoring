# A dtype disagreement on the second of two parameters (#808). The first agrees with its call site, so only the second is reported. The
# supplied signature intentionally disagrees with the call site, so this fixture is analyzed statically rather than executed.
import tensorflow as tf


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ]
)
def f(t, u):
    return t + tf.cast(u, tf.float32)


if __name__ == "__main__":
    f(tf.constant(1.0), tf.constant(2))
