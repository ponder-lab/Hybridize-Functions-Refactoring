import tensorflow as tf


def zeta(x):
    return tf.reduce_sum(x)


class Model:
    def beta(self, x):
        return zeta(x)

    def omega(self, x):
        return tf.reduce_max(x)


def alpha(x):
    return Model().beta(x)


def kappa(x):
    return tf.reduce_min(x)


def delta(x):
    return tf.reduce_mean(x)


def mu(x):
    return tf.reduce_prod(x)


def gamma(x):
    return tf.abs(x)


xs = tf.ones(2)
for f in (zeta, alpha, kappa, delta, mu, gamma):
    f(xs)
Model().omega(xs)
