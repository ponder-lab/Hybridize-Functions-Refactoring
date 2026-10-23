import tensorflow as tf


# Receives the concatenation of one column of `rows`, regrouped by `zip(*rows)`.
def total(column):
    return tf.reduce_sum(column)


rows = [[tf.ones((2, 3)), tf.zeros((2, 3))], [tf.ones((2, 3)), tf.zeros((2, 3))]]

for column in list(zip(*rows)):
    total(tf.concat(list(column), axis=1))
