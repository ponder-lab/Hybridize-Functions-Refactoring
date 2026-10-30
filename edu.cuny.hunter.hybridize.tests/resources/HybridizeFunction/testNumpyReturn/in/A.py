import sys

import numpy as np
import tensorflow as tf


def get_wh_from_boxes(boxes):
    # Fed an ndarray, so eagerly the result is an ndarray, which the caller indexes with an integer
    # array. Traced, the result is a tensor, and that indexing raises.
    return boxes[..., 2:4] - boxes[..., 0:2]


def mixed_wh(boxes):
    # Called once with an ndarray and once with a tensor, so one caller gets an ndarray today.
    return boxes[..., 2:4] - boxes[..., 0:2]


def tensor_wh(boxes):
    # Fed a tensor, so the result is a tensor eagerly too.
    return boxes[..., 2:4] - boxes[..., 0:2]


def summed_wh(boxes):
    # A TensorFlow op makes the result a tensor eagerly, whatever the argument.
    return tf.reduce_sum(boxes[..., 2:4] - boxes[..., 0:2], axis=-1)


def either_wh(boxes, summed):
    # One path returns the array it is passed, the other a TensorFlow result, so the caller taking
    # the first path gets an ndarray today.
    if summed:
        return tf.reduce_sum(boxes, axis=-1)
    return boxes[..., 2:4] - boxes[..., 0:2]


def stacked_wh(boxes):
    # Fed an `np.vstack` result, an ndarray at run time.
    return boxes[..., 2:4] - boxes[..., 0:2]


boxes = np.array([[0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 4.0, 5.0], [2.0, 0.0, 3.0, 1.0]])
picks = np.array([0, 2])

wh = get_wh_from_boxes(boxes)
print(wh[picks])

print(mixed_wh(boxes)[picks])
print(mixed_wh(tf.constant(boxes)))

print(tensor_wh(tf.constant(boxes)))
print(summed_wh(boxes))

print(either_wh(boxes, len(sys.argv) > 1)[picks])

stacked = stacked_wh(np.vstack([boxes, boxes]))
print(stacked[picks])
