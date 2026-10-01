# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/998

import pickle
import unittest

import tensorflow as tf

case = unittest.TestCase()


def mixed(inputs, sequence_length):
    return tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1])


def control(inputs, sequence_length):
    return tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1])


def guarded(x):
    return tf.reduce_sum(x)


def mixed_dtypes(x):
    return tf.reduce_sum(x)


# A test-like call with small, concrete tensors.
mixed(tf.zeros([4, 5, 10], tf.float64), tf.constant([1, 2, 3, 4]))
control(tf.zeros([4, 5, 10], tf.float64), tf.constant([1, 2, 3, 4]))

# The program's own call, whose arguments come from a source the analysis does not model.
with open("batch.pkl", "rb") as f:
    batch = pickle.load(f)

mixed(batch["inputs"], batch["length"])

# A declared failure passing a non-tensor is set aside, so it leaves no untyped context.
guarded(tf.ones([2, 2]))

with case.assertRaises(ValueError):
    guarded(None)

# Typed calls that disagree in dtype keep their own reason beside an untyped call.
mixed_dtypes(tf.ones([3], tf.float32))
mixed_dtypes(tf.ones([3], tf.int32))
mixed_dtypes(batch["other"])
