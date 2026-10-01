# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/998

import pickle

import tensorflow as tf


def mixed(inputs, sequence_length):
    return tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1])


def control(inputs, sequence_length):
    return tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1])


# A test-like call with small, concrete tensors.
mixed(tf.zeros([4, 5, 10], tf.float64), tf.constant([1, 2, 3, 4]))
control(tf.zeros([4, 5, 10], tf.float64), tf.constant([1, 2, 3, 4]))

# The program's own call, whose arguments come from a source the analysis does not model.
with open("batch.pkl", "rb") as f:
    batch = pickle.load(f)

mixed(batch["inputs"], batch["length"])
