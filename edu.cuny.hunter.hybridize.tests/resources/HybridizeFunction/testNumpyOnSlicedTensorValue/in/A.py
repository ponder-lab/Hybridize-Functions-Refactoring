import numpy as np
import tensorflow as tf


def clip_slice(boxes):
    return np.maximum(boxes[:2], 0.0)


b = tf.ones((5, 4))
clip_slice(b)
