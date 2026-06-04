# `tf.constant(np.ones(..., dtype=...))` propagates the numpy array's shape and dtype
# (https://github.com/wala/ML/issues/539, fixed in Ariadne 0.47.0), so `consume`'s parameter types as a concrete
# FLOAT32 (2, 3) tensor and `inferInputSignature` produces a signature. Regression guard that np.ones/np.zeros origins
# are typed, not collapsed to ⊤.
import numpy as np
import tensorflow as tf


def consume(x):
    return x


if __name__ == "__main__":
    consume(tf.constant(np.ones((2, 3), dtype=np.float32)))
