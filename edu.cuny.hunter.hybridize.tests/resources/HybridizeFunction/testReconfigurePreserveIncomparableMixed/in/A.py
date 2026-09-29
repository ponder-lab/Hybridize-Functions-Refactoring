# A dtype disagreement whose call sites pass both a tensor and a NumPy array (#808). The tensor call raises, while the array is silently
# cast, so the warning says both. The supplied signature intentionally disagrees with the call sites, so this fixture is analyzed
# statically rather than executed.
import numpy as np
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(np.array(2))
    f(tf.constant(2, dtype=tf.int64))
