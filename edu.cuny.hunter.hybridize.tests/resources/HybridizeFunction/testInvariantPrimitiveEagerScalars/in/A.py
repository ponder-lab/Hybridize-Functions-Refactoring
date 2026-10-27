# Issue 1109, reduced from tf-eager-fasterrcnn's `bbox2delta` (`detection/core/bbox/transforms.py`), with scalar instance state in
# place of its tuples. The eager function's primitive `std` is set once in each target generator's constructor from the same
# literal, and both callers pass that attribute, so it's one constant at every call and can't vary the trace key. `offset` is a
# control: its two callers pass different constants, so it traces once per value.
import tensorflow as tf


def bbox2delta(box, gt_box, std):
    delta = tf.cast(gt_box, tf.float32) - tf.cast(box, tf.float32)
    return delta / std


def offset(box, amount):
    return tf.cast(box, tf.float32) + amount


class AnchorTarget:

    def __init__(self, std=0.1):
        self.std = std

    def build_single_target(self, anchors, gt):
        return offset(bbox2delta(anchors, gt, self.std), 1.0)


class ProposalTarget:

    def __init__(self, std=0.1):
        self.std = std

    def build_single_target(self, rois, gt):
        return offset(bbox2delta(rois, gt, self.std), 2.0)


anchor_target = AnchorTarget()
proposal_target = ProposalTarget()

for _ in range(3):
    anchor_target.build_single_target(tf.ones([8, 4]), tf.ones([8, 4]))
    proposal_target.build_single_target(tf.ones([8, 4]), tf.ones([8, 4]))
