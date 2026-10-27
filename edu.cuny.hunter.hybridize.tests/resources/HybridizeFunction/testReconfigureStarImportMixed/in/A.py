# Issue 1018: in a file that reaches TensorFlow only through a star import, an eager function converted to hybrid and an
# already-hybrid one reconfigured share a single injected import.
from B import *


def scale(x):
    return x * 2


@tf.function(experimental_relax_shapes=True)
def shift(y):
    return y + 1


if __name__ == "__main__":
    scale(tf.constant([1.0, 2.0]))
    shift(tf.constant([[1, 2]]))
