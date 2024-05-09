import tensorflow as tf


@tf.function(experimental_compile=True)
def add(a, b):
    return a + b


a = tf.range(3, 18, 3)
b = tf.range(5)
c = add(a, b)
