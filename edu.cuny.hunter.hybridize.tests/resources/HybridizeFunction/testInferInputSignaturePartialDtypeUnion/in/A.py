import json

import tensorflow as tf


# A dtype looked up by a name parsed at run time, which the analysis cannot read.
def runtime_dtype(name):
    return getattr(tf, json.loads(name))


# Joins an int32 allocation and one of unreadable dtype in one frame.
def operand(flag, first, second):
    if flag:
        return tf.ones([5], dtype=first)
    return tf.ones([5], dtype=second)


# Receives a value whose dtype is int32 at one allocation and unknown at the other.
def increment(n):
    return n + 1


flag = len(json.loads("[]")) == 0
increment(tf.reduce_max(operand(flag, tf.int32, runtime_dtype('"int32"'))))
