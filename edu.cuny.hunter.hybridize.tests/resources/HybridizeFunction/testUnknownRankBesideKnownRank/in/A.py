# Reduced from tf-image's `_clip_random_with_bboxes` (issue 1150): one call passes a NumPy array whose
# shape is unresolved, and the others pass rank-2 boxes, so the reduced spec is `shape=None`.
# The boxes are returned, and a traced caller mapping over them fails on an empty batch.

import numpy as np
import tensorflow as tf


def clip_boxes(bboxes):
    return bboxes / 2.0


def known_only(bboxes):
    return bboxes / 2.0


def unknown_only(bboxes):
    return bboxes / 2.0


def ranks_disagree(bboxes):
    return bboxes / 2.0


unresolved = np.asarray([[0.1, 0.2, 0.3, 0.4]]).astype(np.float32)

clip_boxes(tf.zeros((0, 4)))
clip_boxes(tf.zeros((1, 4)))
clip_boxes(unresolved)

known_only(tf.zeros((0, 4)))
known_only(tf.zeros((1, 4)))

unknown_only(unresolved)

ranks_disagree(tf.zeros((0, 4)))
ranks_disagree(tf.zeros((4,)))
ranks_disagree(unresolved)
