# Reduced from TensorFlow2.0-Examples, 4-Object_Detection/YOLOV3/core/yolov3.py and train.py. The predicted boxes come from an
# element of the result of a functional model built with a list of outputs, and the label boxes from an `np.zeros` whose shape
# tuple holds an element of an ndarray, as core/dataset.py builds them.
import numpy as np
import tensorflow as tf


def decode(conv_output):
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    return tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 7))


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat(
        [
            boxes1[..., :2] - boxes1[..., 2:] * 0.5,
            boxes1[..., :2] + boxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    boxes2 = tf.concat(
        [
            boxes2[..., :2] - boxes2[..., 2:] * 0.5,
            boxes2[..., :2] + boxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat(
        [
            boxes1[..., :2] - boxes1[..., 2:] * 0.5,
            boxes1[..., :2] + boxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    boxes2 = tf.concat(
        [
            boxes2[..., :2] - boxes2[..., 2:] * 0.5,
            boxes2[..., :2] + boxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    return iou - 1.0 * (enclose_area - union_area) / enclose_area


def compute_loss(pred, label, bboxes):
    pred_xywh = pred[:, :, :, :, 0:4]
    label_xywh = label[:, :, :, :, 0:4]
    giou = bbox_giou(pred_xywh, label_xywh)
    iou = bbox_iou(
        pred_xywh[:, :, :, :, np.newaxis, :],
        bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :],
    )
    return giou, iou


input_tensor = tf.keras.layers.Input([26, 26, 3])
conv_tensor = tf.keras.layers.Conv2D(21, 1, strides=2)(input_tensor)
model = tf.keras.Model(input_tensor, [conv_tensor, decode(conv_tensor)])

train_output_sizes = 26 // np.array([2, 4])
label = np.zeros(
    (4, train_output_sizes[0], train_output_sizes[0], 3, 7), dtype=np.float32
)
bboxes = np.zeros((4, 5, 4), dtype=np.float32)

pred_result = model(tf.ones([4, 26, 26, 3]))
pred = pred_result[1]
assert pred.shape == (4, 13, 13, 3, 7) and label.shape == (4, 13, 13, 3, 7)
giou, iou = compute_loss(pred, label, bboxes)
assert giou.shape == (4, 13, 13, 3) and iou.shape == (4, 13, 13, 3, 5)
