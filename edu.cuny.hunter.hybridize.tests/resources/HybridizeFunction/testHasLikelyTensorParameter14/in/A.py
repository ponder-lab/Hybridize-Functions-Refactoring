import tensorflow as tf

import timeit

conv_layer = tf.keras.layers.Conv2D(100, 3)


# Per https://www.tensorflow.org/guide/function#usage, conv_*can* have tf.function but doesn't benefit from it in this example because there are a few "large" ops.
# But, when I tried it, I get a small speedup.
# @tf.function
def conv_fn(image):
    return conv_layer(image)


image = tf.zeros([1, 200, 200, 100])
# Warm up
conv_layer(image)
conv_fn(image)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("Note how there's not much difference in performance for convolutions")
