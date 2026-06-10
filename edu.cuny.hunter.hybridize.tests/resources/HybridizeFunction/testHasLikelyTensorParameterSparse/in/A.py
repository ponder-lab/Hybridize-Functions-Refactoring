# Test for #434. A `tf.SparseTensor` type hint must classify the parameter as tensor-typed via the Phase 1 type-hint path. The
# argument is a non-tensor so the only classifying signal is the type hint.
import tensorflow as tf


@tf.function
def f(t: tf.SparseTensor):
    pass


if __name__ == "__main__":
    f(5)
