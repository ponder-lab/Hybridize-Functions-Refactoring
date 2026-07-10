import numpy as np
import tensorflow as tf


def data_generator():
    while True:
        scores = tf.ones((1, 4, 2))
        masks = tf.ones((1, 4))
        yield scores, masks


def compute_loss(scores, masks):
    score_loss = tf.reduce_sum(scores)
    weight = np.sum(np.abs(masks))
    return score_loss / weight


generator = data_generator()
s, m = next(generator)
compute_loss(s, m)
