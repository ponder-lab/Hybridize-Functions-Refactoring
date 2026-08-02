# Modify path (#596), supplied-tighter case. The decorator pins a concrete rank-1 shape, but the call sites pass tensors of
# differing rank (scalar and rank-1), so inference degrades the shape to unknown rank (shape=None). The supplied signature is strictly
# tighter than the evidence, so it is overwritten (informational). The supplied signature intentionally disagrees with the call sites,
# so this fixture is analyzed statically rather than executed.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(2,), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
    f(tf.ones([2]))
