# https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/image/extract_patches#example

import tensorflow as tf


def f(a):
    pass


n = 10

# images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100
images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]

# We generate two outputs as follows:
# 1. 3x3 patches with stride length 5
# 2. Same as above, but the rate is increased to 2
r = tf.image.extract_patches(images=images,
                       sizes=[1, 3, 3, 1],
                       strides=[1, 5, 5, 1],
                       rates=[1, 1, 1, 1],
                       padding='VALID')

f(r)
