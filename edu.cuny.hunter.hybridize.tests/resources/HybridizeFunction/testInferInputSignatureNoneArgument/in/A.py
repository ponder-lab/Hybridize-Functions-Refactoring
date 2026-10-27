# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1055

import unittest

import tensorflow as tf

case = unittest.TestCase()


def maybe_none(x, n):
    if n is None:
        return x
    return x * tf.cast(n, x.dtype)


def control(x, n):
    return x * tf.cast(n, x.dtype)


def guarded(x, n):
    return x * tf.cast(n, x.dtype)


# Forwards its parameter, as OpenNMT-tf's `dynamic_decode` forwards `attention_size` to `BeamSearch.initialize`.
def forward(x, n):
    return maybe_none(x, n)


# One argument that is a tensor or `None` by a condition, as OpenNMT-tf's decoder passes `attention_size`.
def call(x, aligned):
    return forward(x, tf.constant(2) if aligned else None)


# OpenNMT-tf's route: decoder.py:458 passes `tf.shape(self.memory)[1] if ... else None` by keyword to `dynamic_decode`, which
# forwards it by keyword to `BeamSearch.initialize`.
def initialize(x, attention_size=None):
    if attention_size is None:
        return x
    return x * tf.cast(attention_size, x.dtype)


def dynamic_decode(x, attention_size=None):
    return initialize(x, attention_size=attention_size)


def decode(memory, x, support_alignment_history):
    return dynamic_decode(
        x, attention_size=tf.shape(memory)[1] if support_alignment_history else None
    )


x = tf.constant([1.0, 2.0])
call(x, True)
call(x, False)

# The same tensor argument, never `None`.
control(x, tf.constant(2))

# A declared failure passing `None` is set aside, so it leaves the specification from the tensor call standing.
guarded(x, tf.constant(2))

with case.assertRaises(ValueError):
    guarded(x, None)

memory = tf.zeros([2, 3])
decode(memory, x, True)
decode(memory, x, False)
