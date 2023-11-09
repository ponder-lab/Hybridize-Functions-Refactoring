import tensorflow as tf


def f():

    @tf.function
    def g():
        return 5

    a = g()
    assert a == 5


f()
