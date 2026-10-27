import tensorflow as tf


# Receives a row of `targets`, the int32 component `call` unpacks and passes on.
def masked(row):
    return tf.boolean_mask(row, tf.not_equal(row, 0))


def build_targets(targets):
    for i in range(2):
        masked(targets[i])


# Receives the nonzero entries of a row of the int32 component `call` unpacks.
def labels(row):
    return tf.reduce_sum(row)


class Model(tf.keras.Model):

    def call(self, inputs):
        images, gt = inputs
        for i in range(2):
            labels(tf.boolean_mask(gt[i], tf.not_equal(gt[i], 0)))
        build_targets(gt)
        return images


model = Model()
model((tf.ones((2, 8, 8, 3)), tf.ones((2, 5), dtype=tf.int32)))
