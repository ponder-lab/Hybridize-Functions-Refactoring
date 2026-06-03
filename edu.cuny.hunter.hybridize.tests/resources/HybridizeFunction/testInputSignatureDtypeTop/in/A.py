# Pinning fixture for the dtype-⊤ drop. `tf.constant(np.ones(..., dtype=...))` currently loses the numpy dtype
# (https://github.com/wala/ML/issues/539), so Ariadne infers a single full-⊤ `TensorType(UNKNOWN, null)` for `consume`'s
# parameter. `inferSpec` drops on dtype-⊤ and the refactoring surfaces an INFO. When that issue is fixed, the dtype becomes
# concrete and the test inverts.
import numpy as np
import tensorflow as tf


def consume(x):
    return x


if __name__ == "__main__":
    consume(tf.constant(np.ones((2, 3), dtype=np.float32)))
