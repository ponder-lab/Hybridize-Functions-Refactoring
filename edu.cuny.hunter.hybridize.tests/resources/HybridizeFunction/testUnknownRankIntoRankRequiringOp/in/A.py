# Reduced from TensorFlow-Examples' image_transformation.py (issue 1150): a NumPy array reaches the
# parameter with its shape unresolved, so its spec would be `shape=None`, and `tf.image.resize`
# raises "'images' contains no shape" when traced under it.

import numpy as np
import tensorflow as tf


def resize_image(image):
    image = tf.image.resize(image, size=(256, 256), preserve_aspect_ratio=False)
    image = tf.cast(image, tf.uint8)
    return image


def crop_image(image):
    return tf.image.central_crop(image=image, central_fraction=0.5)


def cast_only(image):
    return tf.cast(image, tf.float32)


def resize_ranked(image):
    return tf.image.resize(image, size=(4, 4))


img = tf.zeros((8, 8, 3), dtype=tf.uint8)
transformed_img = np.asarray(np.zeros((8, 8, 3))).astype(np.uint8)

resize_image(transformed_img)
crop_image(transformed_img)
cast_only(transformed_img)
resize_ranked(img)
