import tensorflow as tf


def reshape_to(y, n):
    return tf.reshape(y, [n, -1])


def forward(y, n):
    return reshape_to(y, n)


a = tf.ones([2, 6])
b = tf.ones([3, 4])
c = tf.ones([3, 5])
print(forward(a, 2).shape)
print(forward(b, tf.shape(c)[0]).shape)
