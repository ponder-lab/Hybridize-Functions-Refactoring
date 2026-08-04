# Runtime-pinned emission fixture (#810), unanimous-union control for the wildcard case. Parameter `x` is reached by two
# call sites whose concrete argument types were observed by tracing this file under `python3.10`/TF 2.9.3: both
# (256, 784) `float32`. The `assert` statements pin those observations and execute under the harness's `runInput` hook.
# Every dimension and the dtype are unanimous, so the expected emitted signature is fully concrete:
# `[tf.TensorSpec(shape=(256, 784), dtype=tf.float32)]`. Executing the expected `out/A.py` has TF enforce that exact
# shape at both call sites, so a signature drifting away from the observed types fails at runtime, not just textually.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(256, 784), dtype=tf.float32)])
def f(x):
    return x + 1


if __name__ == "__main__":
    batch_x = tf.ones((256, 784), dtype=tf.float32)
    assert tuple(batch_x.shape) == (256, 784) and batch_x.dtype == tf.float32
    f(batch_x)
    batch_y = tf.ones((256, 784), dtype=tf.float32)
    assert tuple(batch_y.shape) == (256, 784) and batch_y.dtype == tf.float32
    f(batch_y)
