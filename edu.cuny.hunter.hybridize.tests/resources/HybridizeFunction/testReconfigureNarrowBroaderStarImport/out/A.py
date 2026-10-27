# Issue 1018, narrowing path (#808): the supplied signature is broader than the call site requires, and the file reaches TensorFlow
# only through a star import. One is injected, and the supplied literal is replaced by the inferred one.
from B import *
from tensorflow import function, TensorSpec, float32


@tf.function(input_signature=[TensorSpec(shape=(), dtype=float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
