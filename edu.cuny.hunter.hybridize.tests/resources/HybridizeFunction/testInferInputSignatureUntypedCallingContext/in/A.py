# Calls whose arguments come from an untyped external source, typed by the sidecar beside this file. Cf.
# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/998, whose real argument is computed rather than loaded.

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


# `inner` is reached only through `outer`.
def inner(inputs, lengths):
    return tf.sequence_mask(lengths, maxlen=tf.shape(inputs)[1])


def outer(inputs, lengths):
    return inner(inputs, lengths)


def optional(x, mask=None):
    if mask is None:
        return x
    return x * mask


def leak_inner(x):
    return x * 2


def leak_outer(x):
    return leak_inner(x)


def subscripted(x):
    return tf.reduce_sum(x)


class Encoder(tf.keras.layers.Layer):
    def build_mask(self, inputs, sequence_length=None):
        return tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1])


class Typed(Encoder):
    def call(self, inputs, sequence_length=None):
        return self.build_mask(inputs, sequence_length=sequence_length)


# A test-like call with small, concrete tensors.
mixed(tf.zeros([4, 5, 10], tf.float64), tf.constant([1, 2, 3, 4]))
control(tf.zeros([4, 5, 10], tf.float64), tf.constant([1, 2, 3, 4]))

# The program's own call, whose arguments come from an external dataset that `ariadne-types.json` declares.
with open("batch.pkl", "rb") as f:
    inputs, length, other = pickle.load(f)

mixed(inputs, length)

# A declared failure passing a non-tensor, which a specification raises the same exception for.
guarded(tf.ones([2, 2]))

with case.assertRaises(ValueError):
    guarded(None)

# Calls that disagree in dtype.
mixed_dtypes(tf.ones([3], tf.float32))
mixed_dtypes(tf.ones([3], tf.int32))
mixed_dtypes(other)

# The loaded call reaches `inner` only through `outer`.
outer(tf.zeros([4, 5, 10], tf.float64), tf.constant([1, 2, 3, 4]))
outer(inputs, length)

# An omitted tensor parameter takes its default, which a specification covering it rejects.
optional(tf.ones([2]), tf.ones([2]))
optional(tf.ones([2]))

# Typed calls through a Keras layer's `call`, by keyword, keep their specification.
Typed()(tf.zeros([4, 5, 10]), sequence_length=tf.constant([4, 3, 5, 2]))

# A declared failure one call level up, which also passes `None` to `leak_inner`.
leak_outer(tf.ones([2]))

with case.assertRaises(TypeError):
    leak_outer(None)

# A value read by subscript from a loaded object is not a named binding, so no sidecar entry can type it.
subscripted(tf.ones([2]))

with open("batch.pkl", "rb") as f:
    batch = pickle.load(f)

subscripted(batch["x"])
