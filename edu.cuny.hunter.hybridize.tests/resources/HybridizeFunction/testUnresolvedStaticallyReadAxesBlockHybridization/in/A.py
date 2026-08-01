import tensorflow as tf


def arith(x):
    return tf.range(x.shape[-1] - 1, -1, -1)


def reshape_derived(x):
    padded = tf.pad(x, [[0, 0], [1, 0]])
    return tf.reshape(padded, [-1, padded.shape[1]])


def dynamic_read(x):
    return tf.reshape(x, [-1, tf.shape(x)[1]])


def pinned(x):
    return tf.range(x.shape[-1] - 1, -1, -1)


def sliced(x):
    dims = x.shape[-2:]
    return tf.range(dims[1])


def dims_of(t):
    return t.shape


def extracted(x):
    return tf.range(dims_of(x)[1])


def keras_static(x):
    return tf.range(tf.keras.backend.int_shape(x)[1])


def casted(x):
    return tf.cast(x, x.dtype) * 2.0


a = tf.ones((2, 4))
b = tf.ones((2, 5))

assert arith(a).shape == (4,)
assert arith(b).shape == (5,)
assert reshape_derived(a).shape == (2, 5)
assert reshape_derived(b).shape == (2, 6)
assert dynamic_read(a).shape == (2, 4)
assert dynamic_read(b).shape == (2, 5)
assert pinned(a).shape == (4,)
assert sliced(a).shape == (4,)
assert sliced(b).shape == (5,)
assert extracted(a).shape == (4,)
assert extracted(b).shape == (5,)
assert keras_static(a).shape == (4,)
assert keras_static(b).shape == (5,)
assert casted(a).shape == (2, 4)
assert casted(b).shape == (2, 5)
