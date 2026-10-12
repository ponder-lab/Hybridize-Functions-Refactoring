import tensorflow as tf


@tf.function
def hybrid_tensor(x):
    return x * 2


@tf.function
def hybrid_primitive(n):
    return tf.ones([n])


def eager_tensor(x):
    return tf.reduce_sum(x)


def eager_primitive(n):
    return n + 1


class Model(tf.keras.Model):

    def call(self, x):
        return x + 1


hybrid_tensor(tf.constant([1.0, 2.0]))
hybrid_primitive(3)
eager_tensor(tf.constant([1.0, 2.0]))
eager_primitive(3)
Model()(tf.constant([1.0]))
