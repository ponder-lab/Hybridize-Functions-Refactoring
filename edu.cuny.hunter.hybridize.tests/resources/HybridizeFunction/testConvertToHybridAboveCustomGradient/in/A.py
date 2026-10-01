import tensorflow as tf


@tf.custom_gradient
def log1pexp(x):
    e = tf.exp(x)

    def grad(upstream):
        return upstream * (1 - 1 / (1 + e))

    return tf.math.log(1 + e), grad


x = tf.constant(100.0)
y = log1pexp(x)
