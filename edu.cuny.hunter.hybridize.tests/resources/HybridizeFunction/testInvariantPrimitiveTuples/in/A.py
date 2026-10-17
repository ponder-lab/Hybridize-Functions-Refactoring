# Issue 1109, reduced from tf-eager-fasterrcnn's `bbox2delta` (`detection/core/bbox/transforms.py`). The eager function's tuple
# arguments `target_means` and `target_stds` are instance state that each target generator sets once in its constructor from tuple
# literals the detector fixes. A tuple of constants is keyed by its contents, so an unchanging tuple adds nothing to the trace key,
# but P1 still declines the function under HAS_PRIMITIVE_PARAMETERS: a known lost conversion (wala/ML#805). `scale_boxes` takes a
# nested tuple of constants.
import tensorflow as tf


def bbox2delta(box, gt_box, target_means, target_stds):
    target_means = tf.constant(target_means, dtype=tf.float32)
    target_stds = tf.constant(target_stds, dtype=tf.float32)
    delta = tf.cast(gt_box, tf.float32) - tf.cast(box, tf.float32)
    return (delta - target_means) / target_stds


def scale_boxes(box, scales):
    return tf.cast(box, tf.float32) * tf.constant(scales, dtype=tf.float32)[0]


class AnchorTarget:

    def __init__(
        self, target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(0.1, 0.1, 0.2, 0.2)
    ):
        self.target_means = target_means
        self.target_stds = target_stds

    def build_single_target(self, anchors, gt):
        return bbox2delta(anchors, gt, self.target_means, self.target_stds)


class ProposalTarget:

    def __init__(
        self, target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(0.1, 0.1, 0.2, 0.2)
    ):
        self.target_means = target_means
        self.target_stds = target_stds

    def build_single_target(self, rois, gt):
        return bbox2delta(rois, gt, self.target_means, self.target_stds)


class FasterRCNN:

    def __init__(self):
        self.anchor_target = AnchorTarget(
            target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(0.1, 0.1, 0.2, 0.2)
        )
        self.proposal_target = ProposalTarget(
            target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(0.1, 0.1, 0.2, 0.2)
        )


model = FasterRCNN()

for _ in range(3):
    model.anchor_target.build_single_target(tf.ones([8, 4]), tf.ones([8, 4]))
    model.proposal_target.build_single_target(tf.ones([8, 4]), tf.ones([8, 4]))
    scale_boxes(tf.ones([8, 4]), ((1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 2.0, 2.0)))
