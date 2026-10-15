# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1081

import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)])
def analysis(x):
    return x * 2.0


@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)])
def analysis_resolved(x):
    return x * 2.0


def dtype_named(name):
    return getattr(tf, name)


analysis(tf.ones([1, 4, 1]))
analysis(tf.ones([1, 4, 1], dtype=dtype_named("float32")))
analysis_resolved(tf.ones([1, 4, 1]))
