# Negative test for #434. A `tf.RaggedTensorSpec` type hint must NOT classify the parameter as tensor-typed: a *Spec descriptor
# describes a tensor but is not itself a tensor, so its FQN is excluded from the recognized set. The argument is a non-tensor, so
# with the type hint rejected there is no classifying signal.
import tensorflow as tf


@tf.function
def f(t: tf.RaggedTensorSpec):
    pass


if __name__ == "__main__":
    f(5)
