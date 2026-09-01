# A function reached through tf.distribute.Strategy.run receives its arguments
# through a boundary that does not preserve the declared structure, so a signature
# written for it describes a calling convention that will not be used (issue 928).
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(["/cpu:0"])


def train_step(inputs):
    images, labels = inputs
    return tf.reduce_sum(images) + tf.reduce_sum(labels)


def distributed_train_step(dataset_inputs):
    per_replica = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica, axis=None)


distributed_train_step((tf.zeros((8, 4)), tf.zeros((8, 2))))
