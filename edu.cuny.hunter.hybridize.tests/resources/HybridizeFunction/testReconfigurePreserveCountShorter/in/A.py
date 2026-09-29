# A supplied signature declaring fewer parameters than the reachable calls pass (#808). The call passes an argument the signature does not
# declare, which raises at runtime, so the signature disagrees with the call and is left unchanged. The supplied signature intentionally
# disagrees with the call site, so this fixture is analyzed statically rather than executed.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t, u):
    return t + u


if __name__ == "__main__":
    f(tf.constant(2.0), tf.constant(3.0))
