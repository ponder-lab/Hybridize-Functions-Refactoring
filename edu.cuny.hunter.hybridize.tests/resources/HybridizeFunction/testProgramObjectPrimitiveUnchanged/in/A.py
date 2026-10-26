# Issues 1025 and 1109, control. The argument `config` is an object of a program class with a primitive field, allocated once.
# `tf.function` keys such an object by its identity, so its field never causes a retrace, and the parameter is no primitive parameter
# (issue 1048): the hybrid `hybrid_apply` is no longer de-hybridized under P3, and the eager `eager_apply` is converted.
import tensorflow as tf


class Config:

    def __init__(self):
        self.rate = 0.5


@tf.function
def hybrid_apply(x, config):
    return x * config.rate


def eager_apply(x, config):
    return x * config.rate


config = Config()

for _ in range(3):
    hybrid_apply(tf.ones([2]), config)
    eager_apply(tf.ones([2]), config)
