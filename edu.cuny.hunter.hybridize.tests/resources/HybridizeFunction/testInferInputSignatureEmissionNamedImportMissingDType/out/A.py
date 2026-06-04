# Named-import variant where `TensorSpec` is in scope but the dtype constant is not. `from tensorflow import function,
# TensorSpec, constant` makes `function` and `TensorSpec` reachable, but NOT `float32`. Even though analysis classifies `x`
# as a tensor and `inferInputSignature` would produce `[TensorSpec(shape=(), dtype=float32)]`, the source-write must skip
# emission because the `float32` dtype constant is not in scope under this import shape; emitting it unqualified would raise
# `NameError` at runtime. The generated decorator is a bare `@function`.
from tensorflow import function, TensorSpec, constant


@function
def f(x):
    return x + 1


if __name__ == "__main__":
    f(constant(2.0))
