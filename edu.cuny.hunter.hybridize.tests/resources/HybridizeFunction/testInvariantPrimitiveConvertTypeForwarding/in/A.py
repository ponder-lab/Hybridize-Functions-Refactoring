# Issue 1025, reduced from tf-image's `random_pad_to_square` (`tf_image/core/bboxes/resize.py`) under its `@convert_type` decorator
# (`tf_image/core/convert_type_decorator.py`). The primitive `max_extend` is always its default 0.1, but the decorator's wrapper
# forwards it through `*args`, so the call site that reaches the function has an unpacked argument at or before the parameter's slot.
# The positional alignment past an unpack is undetermined (wala/ML#751), so whether the parameter's value set is complete can't be
# shown, and today's behavior stays.
import tensorflow as tf
from functools import wraps


def convert_type(function):
    @wraps(function)
    def wrap(image, *args, **kwargs):
        image = tf.image.convert_image_dtype(image, tf.float32)
        bboxes = tf.cast(args[0], tf.float32)
        return function(image, bboxes, *args[1:], **kwargs)

    return wrap


@tf.function
@convert_type
def random_pad_to_square(image, bboxes, max_extend=0.1):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    max_extend = tf.math.maximum(height, width) // tf.cast(
        100 * max_extend, dtype=tf.int32
    )
    return (
        tf.image.pad_to_bounding_box(
            image, 0, 0, height + max_extend, width + max_extend
        ),
        bboxes,
    )


for _ in range(3):
    random_pad_to_square(tf.ones([4, 4, 3]), tf.constant([[0.0, 0.0, 1.0, 1.0]]))

print(random_pad_to_square.experimental_get_tracing_count())  # 1
