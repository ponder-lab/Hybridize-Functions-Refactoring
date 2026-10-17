from functools import wraps

import tensorflow as tf

# A decorator between `@tf.function` and the definition (#1046). `tf.function` traces the callable
# the decorator returns, so a signature constrains that callable's arguments, while the
# definition's parameters see only what the decorator passes on: here, values cast to `float32`.


def convert_type(function):
    @wraps(function)
    def wrap(image, *args, **kwargs):
        image_type = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)
        bboxes = tf.cast(args[0], tf.float32)
        image, bboxes = function(image, bboxes, *args[1:], **kwargs)
        return tf.image.convert_image_dtype(image, image_type, saturate=True), bboxes

    return wrap


# Already hybrid over the decorator: a `float32` signature would reject the `int32` call below.
@tf.function
@convert_type
def flip(image, bboxes):
    return tf.image.flip_left_right(image), bboxes


# The control: hybrid with nothing in between, so the definition is what is traced.
@tf.function
def flip_direct(image, bboxes):
    return tf.image.flip_left_right(image), bboxes


# Eager beneath the decorator: the added `@tf.function` goes innermost, on the definition, so it
# traces the definition and the cast values are exactly what it receives.
@convert_type
def flip_eager(image, bboxes):
    return tf.image.flip_left_right(image), bboxes


# Already hybrid over `tf.custom_gradient`, which passes its arguments through unchanged.
@tf.function
@tf.custom_gradient
def clip_gradient(x):
    def grad(upstream):
        return tf.clip_by_norm(upstream, 1.0)

    return tf.identity(x), grad


flip(tf.ones([24, 8, 3]), tf.constant([[1, 1, 15, 6]]))
flip(tf.ones([24, 8, 3]), tf.constant([[0.1, 0.2, 0.4, 0.7]]))
flip_direct(tf.ones([24, 8, 3]), tf.constant([[0.1, 0.2, 0.4, 0.7]]))
flip_eager(tf.ones([24, 8, 3]), tf.constant([[1, 1, 15, 6]]))
clip_gradient(tf.constant([1.0, 2.0]))
