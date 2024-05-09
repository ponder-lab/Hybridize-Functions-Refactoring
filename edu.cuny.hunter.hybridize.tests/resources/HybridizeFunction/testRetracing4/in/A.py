# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing.

import tensorflow as tf


@tf.function
def f(x):
    print(x)
    return tf.abs(x)


print(f(1))
print(f(2))  # Slow - compiles new graph

print(f(tf.constant(1)))
print(f(tf.constant(2)))  # Fast - reuses f1
