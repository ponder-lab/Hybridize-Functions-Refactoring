import tensorflow as tf


# Receives the tensor `summed` fills its `None` default in with, never `None` itself.
def segment_sum(x, num_segments):
    return tf.math.unsorted_segment_sum(x, tf.constant([0, 1]), num_segments)


def summed(x, num_segments=None):
    if num_segments is None:
        num_segments = tf.constant(2)
    return segment_sum(x, num_segments)


summed(tf.constant([1.0, 2.0]))
