import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal


# A Keras model taking one tensor and returning one tensor.
class Single(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs):
        return self.dense(inputs)


# A Keras model taking one tensor and returning a tuple.
class PairOut(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs):
        y = self.dense(inputs)
        return y, y + 1.0


# A Keras model taking a tuple and returning one tensor, reading each element by index.
class PairIn(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x = inputs[0]
        w = inputs[1]
        return self.dense(x * w)


# A tuple in, a tuple out, the outputs computed by a helper method.
class PairInOut(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mu_layer = tf.keras.layers.Dense(2)
        self.gmm = tf.keras.layers.Dense(2)

    def encode(self, x, w):
        mu_tilde = self.mu_layer(x * w)
        return mu_tilde, mu_tilde + 1.0

    def call(self, inputs):
        x = inputs[0]
        w = inputs[1]
        mu_tilde, log_sigma2_tilde = self.encode(x, w)
        mu_c = self.gmm(tf.ones([2, 2]))
        return mu_tilde, log_sigma2_tilde, mu_c


def plain(x):
    return x * 2.0


def two(x):
    return x, x * 2.0


def first(pair):
    return pair[0]


class Holder:
    pass


class C:
    def __init__(self):
        self.eps = 1e-10
        self.single = Single()
        self.pair_out = PairOut()
        self.pair_in = PairIn()
        self.model = PairInOut()

    def sink(self, v):
        return np.abs(v)

    def _cluster(self, mu_t, mu, sigma2, phi):
        def f(mu_t, mu, sigma2, phi):
            return np.log(self.eps + phi) + np.log(
                self.eps + multivariate_normal.pdf(mu_t, mean=mu, cov=np.diag(sigma2))
            )

        p = np.array(
            [f(mu_t, mu[i], sigma2[i], phi[i]) for i in np.arange(mu.shape[0])]
        )
        return np.argmax(p, axis=0)

    # (a) The parameter straight to the numpy helper.
    def direct(self, x):
        return self.sink(x)

    # (b) Through a plain Python function and through a TensorFlow op.
    def via_plain(self, x):
        return self.sink(plain(x))

    def via_tf(self, x):
        return self.sink(tf.abs(x))

    # (b') Through a tuple built or unpacked in plain Python.
    def via_plain_tuple_out(self, x):
        a, b = two(x)
        return self.sink(b)

    def via_plain_tuple_in(self, x):
        return self.sink(first((x, x)))

    def via_plain_list_in(self, x):
        return self.sink(first([x, x]))

    def via_local_tuple(self, x):
        p = (x, x)
        return self.sink(p[0])

    # Stores into a dict literal, a dict, a list, and an attribute, each read back.
    def via_dict_literal(self, x):
        d = {"k": x}
        return self.sink(d["k"])

    def via_dict_store(self, x):
        d = {}
        d["k"] = x
        return self.sink(d["k"])

    def via_list_store(self, x):
        l = [1.0, 2.0]
        l[0] = x
        return self.sink(l[0])

    def via_attribute_store(self, x):
        h = Holder()
        h.v = x
        return self.sink(h.v)

    # A container holding nothing from a parameter: no refusal.
    def via_untainted_container(self, x):
        p = (np.ones([2], np.float32), 1.0)
        return self.sink(p[0]) + x

    # (c) Through a Keras model returning one tensor.
    def via_model(self, x):
        return self.sink(self.single(x))

    # (d) Through a Keras model returning a tuple that's unpacked, and through one taking a tuple.
    def via_model_tuple_out(self, x):
        a, b = self.pair_out(x)
        return self.sink(b)

    def via_model_tuple_in(self, x):
        return self.sink(self.pair_in((x, x)))

    # (e) The nested `f` and the comprehension, fed directly and through a model taking and returning tuples.
    def cluster_direct(self, x):
        return self._cluster(x, x, x, x)

    def cluster(self, X_c, W_c=None):
        if W_c is None:
            W_c = np.ones(X_c.shape, np.float32)
        else:
            W_c = W_c.astype(np.float32)
        mu_tilde, log_sigma2_tilde, mu_c = self.model((X_c, W_c))
        return self._cluster(mu_tilde, mu_c, tf.ones([2, 2]), tf.ones([2]))


c = C()
t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
c.direct(t)
c.via_plain(t)
c.via_tf(t)
c.via_plain_tuple_out(t)
c.via_plain_tuple_in(t)
c.via_plain_list_in(t)
c.via_local_tuple(t)
c.via_dict_literal(t)
c.via_dict_store(t)
c.via_list_store(t)
c.via_attribute_store(t)
c.via_untainted_container(t)
c.via_model(t)
c.via_model_tuple_out(t)
c.via_model_tuple_in(t)
c.cluster_direct(np.ones([2, 2], np.float32))
c.cluster(np.ones([2, 2], np.float32))
