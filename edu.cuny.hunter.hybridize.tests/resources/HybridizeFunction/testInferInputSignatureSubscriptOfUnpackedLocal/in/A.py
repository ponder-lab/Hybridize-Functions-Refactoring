import tensorflow as tf


# Receives the first row of the int32 component `first` unpacks from its tuple parameter.
def row_total(row):
    return tf.reduce_sum(row)


def first(inputs):
    a, gt = inputs
    row_total(gt[0])
    return a


# Receives each row of the int32 component `rows` unpacks from its tuple parameter.
def loop_total(row):
    return tf.reduce_sum(row)


def rows(inputs):
    a, gt = inputs
    for i in range(2):
        loop_total(gt[i])
    return a


# Receives each row of the int32 component `call` unpacks from its inputs.
def call_total(row):
    return tf.reduce_sum(row)


class Model(tf.keras.Model):

    def call(self, inputs):
        images, gt = inputs
        for i in range(2):
            call_total(gt[i])
        return images


first((tf.ones(2), tf.ones((3, 4), dtype=tf.int32)))
rows((tf.ones(2), tf.ones((3, 4), dtype=tf.int32)))

model = Model()
model((tf.ones((2, 8, 8, 3)), tf.ones((2, 5), dtype=tf.int32)))
