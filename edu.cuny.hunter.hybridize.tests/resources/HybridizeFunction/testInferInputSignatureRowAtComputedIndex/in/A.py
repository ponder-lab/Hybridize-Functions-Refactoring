import tensorflow as tf


def row_total(r):
    return tf.reduce_sum(r)


def first_total(r):
    return tf.reduce_sum(r)


m = tf.ones([3, 4])

for i in range(3):
    print(row_total(m[i, :]))

print(first_total(m[0, :]))
