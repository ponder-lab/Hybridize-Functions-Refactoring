# Same good-hybrid shape as `testReconfigureBareDecorator`, but with input-signature inference disabled (the suite default). The
# precondition matrix must be unchanged: the function still hits the `HAS_NO_PRIMITIVE_PARAMETERS` failure and selects no
# transformation, proving `RECONFIGURE` is gated behind the flag.
import tensorflow as tf


@tf.function
def f(x):
    return x + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
