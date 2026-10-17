# Issue 1025, reduced from tf-image's `_random_augmentations` (`tf_image/application/tools.py`). The hybrid function's primitive
# `prob_demanding_ops` is always 0.5: the eager wrapper declares that default and forwards it, and no caller passes it. Every call
# supplies the parameter, so the defaulted-unsupplied exemption (#795) doesn't apply. Its value can't vary, so it can't make
# `tf.function` retrace, but P3 still de-hybridizes the function: a known lost conversion (wala/ML#805).
import tensorflow as tf


@tf.function
def _random_augmentations(image, prob_demanding_ops):
    if prob_demanding_ops > 0.4:
        return tf.image.flip_left_right(image)
    return image


def random_augmentations(image, prob_demanding_ops=0.5):
    return _random_augmentations(image, prob_demanding_ops)


for _ in range(3):
    random_augmentations(tf.ones([4, 4, 3]))

print(_random_augmentations.experimental_get_tracing_count())  # 1
