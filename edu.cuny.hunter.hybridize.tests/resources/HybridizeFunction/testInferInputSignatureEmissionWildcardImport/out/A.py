# Wildcard-import variant of `testInferInputSignatureEmission`. `from tensorflow import *` makes both `function` and
# `TensorSpec` (plus the dtype constants) reachable unqualified, so the source-write should emit an unqualified
# `@function(input_signature=[TensorSpec(shape=(), dtype=float32)])` rather than skipping emission because the prefix is
# empty.
from tensorflow import *


@function(input_signature=[TensorSpec(shape=(), dtype=float32)])
def f(x):
    return x + 1


if __name__ == "__main__":
    f(constant(2.0))
