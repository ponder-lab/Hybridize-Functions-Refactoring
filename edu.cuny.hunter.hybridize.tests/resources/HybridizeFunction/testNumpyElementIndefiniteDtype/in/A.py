import json
import sys

import tensorflow as tf


def plural_flags(anchors, img_shape):
    # Passed an array that is int32 or float32.
    y_center = (anchors[:, 2] + anchors[:, 0]) / 2
    return tf.where(y_center <= img_shape[0], 1, 0)


def unknown_flags(anchors, img_shape):
    # Passed an array whose dtype comes from parsed data.
    y_center = (anchors[:, 2] + anchors[:, 0]) / 2
    return tf.where(y_center <= img_shape[0], 1, 0)


metas = tf.constant([[0.0, 0.0, 0.0, 5.0, 5.0]])
anchors = tf.constant([[0.0, 0.0, 10.0, 10.0]])

dtype = tf.int32 if len(sys.argv) > 1 else tf.float32
plural_flags(anchors, tf.cast(metas[..., 3:5], dtype).numpy())
unknown_flags(anchors, tf.constant(json.loads("[5.0, 5.0]")).numpy())
