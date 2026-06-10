# Test for #434. A `tf.RaggedTensor` type hint must classify the parameter as tensor-typed via the Phase 1 type-hint path (honored
# here by the harness's global ALWAYS_FOLLOW_TYPE_HINTS), since a RaggedTensor is a tensor for refactoring purposes. The argument is a
# non-tensor so the only classifying signal is the type hint.
import tensorflow as tf


@tf.function
def f(t: tf.RaggedTensor):
    pass


if __name__ == "__main__":
    f(5)
