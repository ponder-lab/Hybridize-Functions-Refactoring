# Fixture for #1292 (distilling tf-eager-fasterrcnn's `BBoxHead._get_bboxes_single`):
# unpacking a tensor parameter iterates it. Eagerly that works on a tensor or a NumPy
# array, but under tracing the parameter is a symbolic tensor, and iterating it raises
# (OperatorNotAllowedInGraphError), so the conversion must be declined. Unpacking a
# Python tuple or list of tensors, a tensor's static shape, or the list `tf.split`
# returns, and reading a tensor by subscript, iterate no tensor and remain convertible.
import numpy as np
import tensorflow as tf


def tuple_unpack(probs, img_shape):
    H, W = img_shape
    return probs * tf.cast(H * W, tf.float32)


def ndarray_unpack(probs, img_shape):
    # Fed a NumPy array, which crosses the `tf.function` boundary as a tensor.
    H, W = img_shape
    return probs * tf.cast(H * W, tf.float32)


def list_target_unpack(x):
    [a, b] = x
    return a + b


def nested_unpack(x):
    (a, b), c = x
    return a + b + tf.reduce_sum(c)


def starred_unpack(x):
    a, *rest = x
    return a + tf.zeros(())


def derived_unpack(x):
    # The unpacked tensor is computed from the parameter, so it is symbolic too.
    y = x * 2.0
    a, b = y
    return a + b


def shape_op_unpack(x):
    # Unlike the static `x.shape`, `tf.shape(x)` is a tensor computed from the
    # parameter, and is symbolic under tracing.
    h, w = tf.shape(x)
    return tf.reshape(x, [h * w])


def for_target_unpack(xs, y):
    # The loop's targets unpack the loop's element, not `xs`. `xs` is a Python list,
    # which the loop unrolls, but each element is a tensor, symbolic under tracing.
    total = y
    for a, b in xs:
        total = total + a * b
    return total


class BBoxHead:
    def get_bboxes(self, probs, img_shape):
        return self._get_bboxes_single(probs, img_shape)

    def _get_bboxes_single(self, probs, img_shape):
        # The same unpacking in a method, as in tf-eager-fasterrcnn.
        H, W = img_shape
        return probs * tf.cast(H * W, tf.float32)


def pair_unpack(pair):
    # A Python tuple of tensors: unpacking it iterates the tuple, not a tensor.
    a, b = pair
    return a + b


def list_unpack(parts):
    # A Python list of tensors.
    a, b = parts
    return a + b


def for_pairs(pairs, y):
    # Each element is a Python tuple of tensors, which the loop's targets unpack.
    total = y
    for a, b in pairs:
        total = total + a * b
    return total


def subscript_read(x):
    return x[0] + x[1]


def shape_unpack(x):
    # The static shape is a Python value that tracing keeps concrete.
    h, w = x.shape
    return tf.reshape(x, [h * w])


def split_unpack(x):
    # `tf.split` returns a Python list of tensors.
    a, b = tf.split(x, 2)
    return a + b


v = tf.constant([1.0, 2.0])
m = tf.constant([[1.0, 2.0], [3.0, 4.0]])
probs = tf.ones([2])

assert tuple_unpack(probs, tf.constant([5, 4])).numpy().tolist() == [20.0, 20.0]
assert ndarray_unpack(probs, np.array([5, 4], dtype=np.int32)).numpy().tolist() == [
    20.0,
    20.0,
]
assert float(list_target_unpack(v)) == 3.0
assert float(nested_unpack(m)) == 10.0
assert float(starred_unpack(v)) == 1.0
assert float(derived_unpack(v)) == 6.0
assert shape_op_unpack(m).numpy().tolist() == [1.0, 2.0, 3.0, 4.0]
assert float(for_target_unpack([v, v], tf.zeros(()))) == 4.0
assert BBoxHead().get_bboxes(probs, tf.constant([5, 4])).numpy().tolist() == [
    20.0,
    20.0,
]
assert float(pair_unpack((tf.constant(1.0), tf.constant(2.0)))) == 3.0
assert float(list_unpack([tf.constant(1.0), tf.constant(2.0)])) == 3.0
assert float(for_pairs([(tf.constant(1.0), tf.constant(2.0))], tf.zeros(()))) == 2.0
assert float(subscript_read(v)) == 3.0
assert shape_unpack(m).numpy().tolist() == [1.0, 2.0, 3.0, 4.0]
assert split_unpack(v).numpy().tolist() == [3.0]
