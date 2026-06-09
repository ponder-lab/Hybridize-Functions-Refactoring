# Named-import variant: `from tensorflow import function` brings the decorator into scope but not `TensorSpec`, so the inferred
# signature cannot be emitted unqualified. RECONFIGURE must NOT be selected (emission would be a no-op); the function keeps its
# HAS_NO_PRIMITIVE_PARAMETERS status. Mirrors `testInferInputSignatureEmissionNamedImport` for the already-hybrid case.
from tensorflow import function, constant


@function
def f(x):
    return x + 1


if __name__ == "__main__":
    f(constant(2.0))
