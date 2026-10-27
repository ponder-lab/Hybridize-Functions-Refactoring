import tensorflow as tf


# Every parameter is keyword-only, so each one's index among the keyword-only parameters is also its declaration index.
def f(*, x, scale=1):
    return x * scale


f(x=tf.ones([2]), scale=2)
f(x=tf.ones([3]), scale=2)
