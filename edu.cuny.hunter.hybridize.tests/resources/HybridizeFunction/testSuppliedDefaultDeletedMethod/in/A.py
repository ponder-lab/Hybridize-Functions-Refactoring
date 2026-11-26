# Fixture for #1323 over the trampoline path: the method is called bound, through `self` and
# through an instance, by keyword and positionally, each time with its default.
import tensorflow as tf


class Attention:

    def softmax(self, x, axis=-1):
        x = x - tf.reduce_max(x, axis=axis, keepdims=True)
        ex = tf.exp(x)
        return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

    def attend(self, x):
        return self.softmax(x, axis=-1)


attention = Attention()
attention.attend(tf.constant([1.0, 2.0]))
attention.softmax(tf.constant([3.0, 4.0]), -1)
