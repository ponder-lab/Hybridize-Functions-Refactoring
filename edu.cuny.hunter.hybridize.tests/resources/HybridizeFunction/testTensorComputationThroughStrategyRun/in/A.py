import tensorflow as tf


def train_step(inputs):
    return tf.reduce_mean(inputs)


def dist_train_step(strategy, inputs):
    return strategy.run(train_step, args=(inputs,))


strategy = tf.distribute.MirroredStrategy()
result = dist_train_step(strategy, tf.ones([2, 2]))
