# Shape A (#578): a bundled `from tensorflow import function, TensorSpec, float32, constant` makes `function`, `TensorSpec`, and the
# dtype constant all reachable unqualified, so the source-write emits an unqualified `input_signature`. The first imported
# name (`function`) must not short-circuit the scan and skip emission.
from tensorflow import function, TensorSpec, float32, constant


def f(x):
    return x + 1


if __name__ == "__main__":
    f(constant(2.0))
