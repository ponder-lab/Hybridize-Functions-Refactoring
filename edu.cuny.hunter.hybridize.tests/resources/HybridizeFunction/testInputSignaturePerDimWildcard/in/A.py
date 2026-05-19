# Regression fixture for #494 / #507 per-dim wildcard emission. Parameter `t` is reached by two
# call sites with same-rank (rank 1) tensors of differing size at position 0: shape (2,) and
# shape (3,). Per Algorithm 2 step 3, position-0 has |D_0| = 2 (concrete values 2 and 3 disagree),
# so the emitted shape uses `SymbolicDim("?")` at position 0. Dtype consensus: both float32.
# Expected output: TensorType(FLOAT32, [SymbolicDim("?")]).

import tensorflow as tf


def func(t):
    return t


func(tf.constant([1.0, 2.0]))
func(tf.constant([10.0, 20.0, 30.0]))
