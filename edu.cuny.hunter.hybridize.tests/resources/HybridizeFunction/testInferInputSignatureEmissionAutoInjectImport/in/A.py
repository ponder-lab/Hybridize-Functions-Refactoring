# Auto-inject variant (#574): the file has no TensorFlow import at all, but `x` is typed as a tensor via `np.ones` (Ariadne
# types `np.ones`/`np.zeros` as tensors). The source-write injects `from tensorflow import function, TensorSpec, <dtype>` so
# the inferred `input_signature` can be emitted unqualified, rather than injecting a bare `from tensorflow import function`
# and skipping emission for lack of `TensorSpec`/dtype scope.
import numpy as np


def f(x):
    return x + 1


if __name__ == "__main__":
    f(np.ones((2, 3)))
