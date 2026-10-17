import tensorflow as tf


class Config:

    def __init__(self, rate):
        self.rate = rate


class MyAdam(tf.keras.optimizers.Adam):
    pass


class MyAdamInit(tf.keras.optimizers.Adam):

    def __init__(self, rate):
        super().__init__(learning_rate=rate)


def with_optimizer(opt, x):
    return x * 2


def with_program_object(cfg, x):
    return x * 2


def with_optimizer_subclass(opt, x):
    return x * 2


def with_optimizer_subclass_init(opt, x):
    return x * 2


x = tf.constant([1.0, 2.0])
with_optimizer(tf.keras.optimizers.Adam(1e-4), x)
with_program_object(Config(1e-4), x)
with_optimizer_subclass(MyAdam(1e-4), x)
with_optimizer_subclass_init(MyAdamInit(1e-4), x)
