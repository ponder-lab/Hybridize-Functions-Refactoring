import tensorflow as tf


def reads_numpy(x):
    y = x * 2.0
    return {"value": y.numpy}


def calls_numpy(x):
    y = x * 2.0
    return y.numpy()


class Holder:
    def __init__(self):
        self.numpy = 3.0


holder = Holder()


def reads_holder_numpy(x):
    y = x * 2.0
    return y + holder.numpy


def reads_experimental_numpy(x):
    return tf.experimental.numpy.square(x)


t = tf.constant([1.0, 2.0])
assert callable(reads_numpy(t)["value"])
assert (calls_numpy(t) == [2.0, 4.0]).all()
assert (reads_holder_numpy(t).numpy() == [5.0, 7.0]).all()
assert (reads_experimental_numpy(t).numpy() == [1.0, 4.0]).all()
