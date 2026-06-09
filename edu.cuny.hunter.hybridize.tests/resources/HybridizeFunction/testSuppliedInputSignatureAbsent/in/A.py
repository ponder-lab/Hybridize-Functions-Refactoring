import tensorflow as tf


@tf.function
def func(t):
    return t + 1


if __name__ == "__main__":
    func(tf.ones([2, 2]))
