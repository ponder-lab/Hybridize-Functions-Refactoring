# Issue 1025, reduced from tf-image's `random_aspect_ratio_deformation` (`tf_image/core/bboxes/resize.py`). The primitive arguments
# aren't constant overall, but each of the three call sites passes literals (or the defaults), so the traces are bounded by the call
# sites, at most three, and never grow per call. P3 must not de-hybridize the function. `jitter` is reached from a single call site
# in `apply_jitter`, so its two values share one calling context, but they are still a small fixed set: two traces at most.
import tensorflow as tf


@tf.function
def random_aspect_ratio_deformation(image, max_squeeze=0.7, max_stretch=1.3):
    factor = tf.random.uniform([], max_squeeze, max_stretch)
    height = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * factor, tf.int32)
    return tf.image.resize(image, [height, tf.shape(image)[1]])


@tf.function
def jitter(image, amount):
    return image + tf.random.uniform(tf.shape(image), -amount, amount)


def apply_jitter(image, amount):
    return jitter(image, amount)


for _ in range(3):
    random_aspect_ratio_deformation(tf.ones([4, 4, 3]))
    random_aspect_ratio_deformation(
        tf.ones([4, 4, 3]), max_squeeze=0.6, max_stretch=1.3
    )
    random_aspect_ratio_deformation(
        tf.ones([4, 4, 3]), max_squeeze=0.5, max_stretch=1.2
    )
    apply_jitter(tf.ones([4, 4, 3]), 0.1)
    apply_jitter(tf.ones([4, 4, 3]), 0.2)

print(random_aspect_ratio_deformation.experimental_get_tracing_count())  # 3
print(jitter.experimental_get_tracing_count())  # 2
