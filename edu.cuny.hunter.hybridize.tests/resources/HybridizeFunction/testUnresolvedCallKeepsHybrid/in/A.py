import tensorflow as tf


@tf.function
def opaque(x):
    # Computes through an op the analysis can't resolve: the callee is fetched with `getattr`.
    return getattr(tf, "reduce_sum")(x)


@tf.function
def barren(x):
    return x


def opaque_eager(x):
    return getattr(tf, "reduce_sum")(x)


t = tf.constant([1.0, 2.0, 3.0])
opaque(t)
barren(t)
opaque_eager(t)
