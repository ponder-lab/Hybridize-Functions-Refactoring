# Issues 1025 and 1109, control. The primitive `scale` takes five distinct constants, one per call site, so each value is a separate
# trace, more than a small fixed set bounds. The hybrid `hybrid_scale` is still de-hybridized under P3, and the eager `eager_scale`
# is still declined under HAS_PRIMITIVE_PARAMETERS.
import tensorflow as tf


@tf.function
def hybrid_scale(x, scale):
    return x * scale


def eager_scale(x, scale):
    return x * scale


hybrid_scale(tf.ones([2]), 1.0)
hybrid_scale(tf.ones([2]), 2.0)
hybrid_scale(tf.ones([2]), 3.0)
hybrid_scale(tf.ones([2]), 4.0)
hybrid_scale(tf.ones([2]), 5.0)

eager_scale(tf.ones([2]), 1.0)
eager_scale(tf.ones([2]), 2.0)
eager_scale(tf.ones([2]), 3.0)
eager_scale(tf.ones([2]), 4.0)
eager_scale(tf.ones([2]), 5.0)

print(hybrid_scale.experimental_get_tracing_count())  # 5
