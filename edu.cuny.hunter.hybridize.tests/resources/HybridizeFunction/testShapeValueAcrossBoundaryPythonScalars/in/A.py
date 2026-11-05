import tensorflow as tf


def resize(image, height, width):
    return tf.image.resize(image, [height, width])


def reshape_by(y, s):
    return tf.reshape(y, s)


t = tf.ones([1, 4, 4, 3])
print(resize(t, 8, 6).shape)
print(reshape_by(tf.ones([2, 6]), tf.shape(tf.ones([6, 2]))).shape)
