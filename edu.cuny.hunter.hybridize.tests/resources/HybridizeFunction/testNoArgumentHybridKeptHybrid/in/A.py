import tensorflow as tf

x = tf.ones([2, 2])


@tf.function
def train():
    return tf.reduce_sum(x * 2.0)


@tf.function
def constant():
    return 5


class Counter:

    def __init__(self):
        self.total = tf.Variable(0.0)

    @tf.function
    def update(self):
        self.total.assign_add(tf.reduce_sum(x))


@tf.function
def scale(s):
    return 5 * s


train()
constant()
Counter().update()
scale(2)
