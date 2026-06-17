"""Auto-inject union variant (#588): an import-less file with two hybridizable functions whose inferred input signatures need divergent dtypes (`f` -> float64 via `np.ones`, `g` -> int32 via `np.ones(..., dtype=np.int32)`). The single auto-injected `from tensorflow import ...` line must carry the union of both dtypes so each function's `input_signature` emits unqualified, rather than carrying only the first-processed function's dtype and gating the other off emission."""

import numpy as np


def f(x):
    return x + 1


def g(y):
    return y + 1


if __name__ == "__main__":
    f(np.ones((2, 3)))
    g(np.ones((4,), dtype=np.int32))
