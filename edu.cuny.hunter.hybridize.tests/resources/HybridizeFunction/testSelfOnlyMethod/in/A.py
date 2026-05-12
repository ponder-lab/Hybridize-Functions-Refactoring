import tensorflow as tf


class C:

    @tf.function
    def f(self):
        return tf.constant(1)


C().f()
