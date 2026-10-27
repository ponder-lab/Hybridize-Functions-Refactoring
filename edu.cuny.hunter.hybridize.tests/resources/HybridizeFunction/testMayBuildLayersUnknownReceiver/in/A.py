import tensorflow as tf


class Projector:

    def project(self, x):
        return tf.keras.layers.Dense(4)(x)


# Called unbound on a list literal, so its receiver is no object a class constructor allocated.
Projector.project([], tf.ones([2, 4]))
