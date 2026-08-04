# Runtime-pinned emission fixture (#810). Parameter `x` is reached by three call sites whose concrete argument types were
# observed by tracing this file under `python3.10`/TF 2.9.3: shapes (256, 784), (10000, 784), and (5, 784), all `float32`.
# The `assert` statements pin those observations and execute under the harness's `runInput` hook. Dimension 0 disagrees
# across the union and generalizes to a wildcard; dimension 1 and the dtype are unanimous, so the expected emitted
# signature is `[tf.TensorSpec(shape=(None, 784), dtype=tf.float32)]`, which the expected `out/A.py` pins textually and
# enforces when executed.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(None, 784), dtype=tf.float32)])
def f(x):
    return x + 1


if __name__ == "__main__":
    batch_x = tf.ones((256, 784), dtype=tf.float32)
    assert tuple(batch_x.shape) == (256, 784) and batch_x.dtype == tf.float32
    f(batch_x)
    x_test = tf.ones((10000, 784), dtype=tf.float32)
    assert tuple(x_test.shape) == (10000, 784) and x_test.dtype == tf.float32
    f(x_test)
    sample = tf.ones((5, 784), dtype=tf.float32)
    assert tuple(sample.shape) == (5, 784) and sample.dtype == tf.float32
    f(sample)
