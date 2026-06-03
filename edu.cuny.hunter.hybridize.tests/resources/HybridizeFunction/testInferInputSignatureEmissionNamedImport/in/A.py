# Named-import variant of `testInferInputSignatureEmission`. `from tensorflow import function, constant` makes `function`
# reachable but NOT `TensorSpec`. Even though analysis classifies `x` as a tensor and `inferInputSignature` would produce
# `[TensorSpec(shape=(), dtype=float32)]`, the source-write must skip emission because `TensorSpec` is not in scope under
# this import shape; the generated decorator is a bare `@function`.
from tensorflow import function, constant


def f(x):
    return x + 1


if __name__ == "__main__":
    f(constant(2.0))
