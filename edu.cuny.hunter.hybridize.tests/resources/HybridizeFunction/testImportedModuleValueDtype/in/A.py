# Each function is passed a tensor created from an `int`, or from a list of them. The `local_*`
# ones read a value bound in this module, the `imported_*` ones a value bound at the top level
# of `B`, and `imported_call_append` one that a function of `B` returns. At run time a value
# has the same dtype wherever it was bound: int32 from `tf.constant`, int64 from NumPy.
import numpy as np
import tensorflow as tf

import B
from B import PAD

LOCAL_PAD = 388

IDS = np.ones((4,), dtype=np.int64)


def local_constant(x):
    return x


def imported_constant(x):
    return x


def local_array(x):
    return x


def imported_array(x):
    return x


def local_full(x):
    return x


def imported_full(x):
    return x


def local_append(x):
    return x


def imported_append(x):
    return x


def from_imported_append(x):
    return x


def imported_list_append(x):
    return x


def imported_call_append(x):
    return x


local_constant(tf.constant(LOCAL_PAD))
imported_constant(tf.constant(B.PAD))
local_array(tf.constant(np.array(LOCAL_PAD)))
imported_array(tf.constant(np.array(B.PAD)))
local_full(tf.constant(np.full((2,), LOCAL_PAD)))
imported_full(tf.constant(np.full((2,), B.PAD)))
local_append(tf.constant(np.append(IDS, LOCAL_PAD)))
imported_append(tf.constant(np.append(IDS, B.PAD)))
from_imported_append(tf.constant(np.append(IDS, PAD)))
imported_list_append(tf.constant(np.append(IDS, B.PADS)))
imported_call_append(tf.constant(np.append(IDS, B.pad())))
