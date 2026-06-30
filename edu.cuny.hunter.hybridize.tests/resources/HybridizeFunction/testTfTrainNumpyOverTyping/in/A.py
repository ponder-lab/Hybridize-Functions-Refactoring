import numpy as np
import tensorflow as tf


def serialize(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


serialize(np.array([1, 2, 3]))
