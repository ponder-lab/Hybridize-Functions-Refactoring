import tensorflow as tf


def unstack_sum(pair):
    # Receives the list `tf.unstack` returns, and unpacks it.
    a, b = pair
    return a + b


def split_sum(parts):
    # Receives the list `tf.split` returns, and unpacks it.
    a, b = parts
    return a + b


def per_anchor(pred_boxes, target_boxes, kind="iou"):
    # The shape of automl's `_iou_per_anchor`: unpacks lists of coordinate tensors.
    t_ymin, t_xmin, t_ymax, t_xmax = target_boxes
    p_ymin, p_xmin, p_ymax, p_xmax = pred_boxes
    return (t_ymax - t_ymin) * (p_xmax - p_xmin)


def boxes_loss(pred_boxes, target_boxes):
    # The shape of automl's `iou_loss`: unstacks, slices, and rebuilds the slices with a comprehension.
    pred_list = tf.unstack(pred_boxes, None, axis=-1)
    target_list = tf.unstack(target_boxes, None, axis=-1)
    losses = []
    for i in range(0, len(pred_list), 4):
        pred = pred_list[i : i + 4]
        target = target_list[i : i + 4]
        t_ymin, t_xmin, t_ymax, t_xmax = target
        mask = tf.cast(
            tf.math.logical_and(t_ymax > t_ymin, t_xmax > t_xmin), t_ymin.dtype
        )
        pred = [b * mask for b in pred]
        target = [b * mask for b in target]
        losses.append(mask * per_anchor(pred, target, "iou"))
    return losses[0]


def slice_sum(parts):
    # Receives a slice of an unstacked list, and unpacks it.
    a, b, c, d = parts
    return a + b + c + d


def first_of(x):
    # Control: receives one element of an unstacked list.
    return x * 2.0


def split_heads(x):
    # Control: gpt-2's shape, a split unpacked by the caller, one element passed on.
    return x * 2.0


def attend(q):
    k, v = tf.split(q, 2, axis=-1)
    return split_heads(k) + split_heads(v)


unstack_sum(tf.unstack(tf.zeros([2, 3])))
split_sum(tf.split(tf.zeros([4, 3]), 2))
boxes_loss(tf.ones([2, 4]), tf.ones([2, 4]))
first_of(tf.unstack(tf.zeros([2, 3]))[0])
slice_sum(tf.unstack(tf.zeros([2, 8]), axis=-1)[0:4])
attend(tf.zeros([2, 6]))
