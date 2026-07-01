import tensorflow as tf


def f(x):
    return x


print(f(tf.constant(1)))
print(f(tf.constant(2)))
