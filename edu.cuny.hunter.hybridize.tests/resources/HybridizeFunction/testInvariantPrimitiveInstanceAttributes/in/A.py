# Issue 1025, reduced from VaDER's `VaDER/vader.py`. The hybrid loss methods' primitive arguments `K` and `eps` come from instance
# state set once in `__init__` and never reassigned, and every call passes those same attributes, so they can't vary the trace key,
# but P3 still de-hybridizes the methods: a known lost conversion (wala/ML#805). `output_activation` is a function object chosen in
# `__init__`, not a primitive.
import tensorflow as tf


class VaDER:

    def __init__(self, k=3, eps=1e-10, output_activation=None):
        self.eps = eps
        self.K = k
        if output_activation is None:
            self.output_activation = tf.identity
        else:
            if output_activation == "sigmoid":
                self.output_activation = tf.nn.sigmoid

    @tf.function
    def _reconstruction_loss(self, X, x, output_activation, eps=1e-10):
        if output_activation == tf.nn.sigmoid:
            return tf.reduce_mean(tf.clip_by_value(X, eps, 1 - eps) - x)
        return tf.reduce_mean(tf.square(X - x))

    @tf.function
    def _latent_loss(self, z, K, eps=1e-10):
        if K == 1:  # ordinary VAE
            return tf.reduce_mean(tf.square(z)) + eps
        return tf.reduce_mean(z) * K + eps

    def _update_state(self, X, x, z):
        rec_loss = self._reconstruction_loss(X, x, self.output_activation, self.eps)
        latent_loss = self._latent_loss(z, self.K, self.eps)
        return rec_loss + latent_loss

    def get_loss(self, X, x, z):
        rec_loss = self._reconstruction_loss(X, x, self.output_activation, self.eps)
        latent_loss = self._latent_loss(z, self.K, self.eps)
        return rec_loss + latent_loss


vader = VaDER()

for _ in range(3):
    vader._update_state(tf.ones([2, 3]), tf.zeros([2, 3]), tf.ones([2, 2]))

vader.get_loss(tf.ones([2, 3]), tf.zeros([2, 3]), tf.ones([2, 2]))

print(vader._latent_loss.experimental_get_tracing_count())  # 1
print(vader._reconstruction_loss.experimental_get_tracing_count())  # 1
