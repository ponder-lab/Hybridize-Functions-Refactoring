import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def tail_parameter(x, k):
    inner = np.prod(get_shape(x)[-k:])
    return tf.reshape(x, [-1, inner])


def tail_local(x):
    k = 1
    inner = np.prod(get_shape(x)[-k:])
    return tf.reshape(x, [-1, inner])


def tail_inverted(x):
    k = 1
    inner = np.prod(get_shape(x)[~k:])
    return tf.reshape(x, [-1, inner])


def lead_local(x):
    k = 1
    outer = np.prod(get_shape(x)[:k])
    return tf.reshape(x, [outer, -1])


# The second feed's leading axis is graph-time `None` and its last axis is 4. A
# suffix slice of one axis covers only the known one, and a prefix slice of one
# axis, or a suffix slice of two, covers the unknown one.
known = tf.ones((2, 4))
unknown = tf.keras.Input(shape=(4,))


# The feed is picked inside a function body rather than in the module's own body:
# the engine folds a module-level `tf.executing_eagerly()` branch to its eager arm
# (wala/ML#1110), which would leave only `known`, while a function keeps both
# arms. The script itself runs eagerly and so takes the `known` arm.
def drive():
    if tf.executing_eagerly():
        pick = known
    else:
        pick = unknown

    tail_parameter(pick, 1)
    tail_local(pick)
    tail_inverted(pick)
    lead_local(pick)


drive()
