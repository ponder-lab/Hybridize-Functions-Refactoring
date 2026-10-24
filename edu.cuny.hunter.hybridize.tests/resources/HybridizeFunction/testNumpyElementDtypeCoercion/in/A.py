import tensorflow as tf


def calc_img_shapes(meta):
    return tf.cast(meta[..., 3:5], tf.int32).numpy()


def calc_float_shapes(meta):
    return meta[..., 3:5].numpy()


def valid_flags(anchors, img_shape):
    # Eagerly, `img_shape[0]` is a NumPy int32 scalar, which the comparison converts to the
    # other operand's float32. Traced, `img_shape` is an int32 tensor, and the comparison raises.
    y_center = (anchors[:, 2] + anchors[:, 0]) / 2
    return tf.where(y_center <= img_shape[0], 1, 0)


def same_dtype_flags(anchors, img_shape):
    # Fed a float32 row, so the element has the other operand's dtype, traced or not.
    y_center = (anchors[:, 2] + anchors[:, 0]) / 2
    return tf.where(y_center <= img_shape[0], 1, 0)


def tensor_flags(anchors, img_shape):
    # Fed a tensor row, which is a tensor eagerly too.
    y_center = (anchors[:, 2] + anchors[:, 0]) / 2
    return tf.where(y_center <= tf.cast(img_shape[0], tf.float32), 1, 0)


def proposals(probs, img_shape):
    # Unpacking iterates `img_shape`, which raises once it is a symbolic tensor.
    H, W = img_shape
    return probs / tf.constant([H, W], dtype=tf.float32)


class AnchorGenerator:
    def generate(self, anchors, img_metas):
        img_shapes = calc_img_shapes(img_metas)
        return self._generate_valid_flags(anchors, img_shapes[0])

    def _generate_valid_flags(self, anchors, img_shape):
        # The same comparison in a method, as in tf-eager-fasterrcnn.
        y_center = (anchors[:, 2] + anchors[:, 0]) / 2
        return tf.where(y_center <= img_shape[0], 1, 0)


metas = tf.constant([[0.0, 0.0, 0.0, 5.0, 5.0]])
anchors = tf.constant([[0.0, 0.0, 10.0, 10.0]])

img_shapes = calc_img_shapes(metas)
float_shapes = calc_float_shapes(metas)
tensor_shapes = tf.cast(metas[..., 3:5], tf.int32)

valid_flags(anchors, img_shapes[0])
same_dtype_flags(anchors, float_shapes[0])
tensor_flags(anchors, tensor_shapes[0])
proposals(tf.ones([2]), img_shapes[0])
AnchorGenerator().generate(anchors, metas)
