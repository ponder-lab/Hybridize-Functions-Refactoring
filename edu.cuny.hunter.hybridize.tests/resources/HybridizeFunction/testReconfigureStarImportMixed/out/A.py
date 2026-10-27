# Issue 1018: in a file that reaches TensorFlow only through a star import, an eager function converted to hybrid and an
# already-hybrid one reconfigured share a single injected import.
from B import *
from tensorflow import function, TensorSpec, float32, int32


@function(input_signature=[TensorSpec(shape=(2,), dtype=float32)])
def scale(x):
    return x * 2


@tf.function(experimental_relax_shapes=True, input_signature=[TensorSpec(shape=(1, 2), dtype=int32)])
def shift(y):
    return y + 1


if __name__ == "__main__":
    scale(tf.constant([1.0, 2.0]))
    shift(tf.constant([[1, 2]]))
