import tensorflow as tf

SIZES = [0] * 12


def coarsen(x, assignment):
    num_clusters = tf.shape(assignment)[-1]
    return tf.reshape(tf.tile(x, [num_clusters, 1]), -1)


def pool(x, assignment):
    return coarsen(x, assignment) * 2.0


def flatten_by_keyword(x):
    return tf.reshape(x, shape=-1)


def flatten_by_scalar_tensor(x):
    return tf.reshape(x, tf.constant(-1))


def flatten_by_vector(x):
    return tf.reshape(x, [-1])


def flatten_by_computed_int(x):
    return tf.reshape(x, x.shape[0] * x.shape[1])


def flatten_by_len(x):
    return tf.reshape(x, len(SIZES))


def flatten_by_starred(x):
    args = (x, -1)
    return tf.reshape(*args)


t = tf.zeros([3, 4])
a = tf.ones([5, 2])
print(pool(t, a).shape)
print(flatten_by_keyword(t).shape)
print(flatten_by_scalar_tensor(t).shape)
print(flatten_by_vector(t).shape)
print(flatten_by_computed_int(t).shape)
print(flatten_by_len(t).shape)
print(flatten_by_starred(t).shape)
