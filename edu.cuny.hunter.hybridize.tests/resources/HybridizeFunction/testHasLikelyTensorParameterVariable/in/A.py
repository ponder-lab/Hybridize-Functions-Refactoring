# Test for #434. A `tf.Variable` type hint must classify the parameter as tensor-typed via the Phase 1 type-hint path; a Variable is
# treated as tensor-equivalent for hybridization purposes. The argument is a non-tensor so the only classifying signal is the hint.
import tensorflow as tf


@tf.function
def f(t: tf.Variable):
    pass


if __name__ == "__main__":
    f(5)
