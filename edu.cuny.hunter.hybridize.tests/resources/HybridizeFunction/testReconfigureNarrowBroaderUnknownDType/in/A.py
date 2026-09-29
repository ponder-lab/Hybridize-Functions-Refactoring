# A broader supplied signature spelling a dtype TensorFlow does not define (#808). The parser cannot read it, so it does not model the
# signature, which is never narrowed. The dtype does not exist, so this fixture is analyzed statically rather than executed.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.unknown)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
