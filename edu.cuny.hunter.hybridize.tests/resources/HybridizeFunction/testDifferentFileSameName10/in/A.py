import tensorflow as tf


@tf.function
def b(a):
    pass


if __name__ == '__main__':
    b(tf.ones([1, 2]))
