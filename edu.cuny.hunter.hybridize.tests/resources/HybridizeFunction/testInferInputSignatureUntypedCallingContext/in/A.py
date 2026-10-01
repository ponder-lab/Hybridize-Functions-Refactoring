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


# `inner` has one call-graph node, shared by both of `outer`'s, so only the caller's argument shows the untyped call.
def inner(inputs, lengths):
    return tf.sequence_mask(lengths, maxlen=tf.shape(inputs)[1])


def outer(inputs, lengths):
    return inner(inputs, lengths)


def optional(x, mask=None):
    if mask is None:
        return x
    return x * mask


class Encoder(tf.keras.layers.Layer):
    def build_mask(self, inputs, sequence_length=None):
        return tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1])


class Typed(Encoder):
    def call(self, inputs, sequence_length=None):
        return self.build_mask(inputs, sequence_length=sequence_length)


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

# The untyped call reaches `inner` only through `outer`.
outer(tf.zeros([4, 5, 10], tf.float64), tf.constant([1, 2, 3, 4]))
outer(batch["inputs"], batch["length"])

# An omitted tensor parameter takes its default, which a specification covering it rejects.
optional(tf.ones([2]), tf.ones([2]))
optional(tf.ones([2]))

# Typed calls through a receiver trampoline and a keyword argument keep their specification.
Typed()(tf.zeros([4, 5, 10]), sequence_length=tf.constant([4, 3, 5, 2]))
