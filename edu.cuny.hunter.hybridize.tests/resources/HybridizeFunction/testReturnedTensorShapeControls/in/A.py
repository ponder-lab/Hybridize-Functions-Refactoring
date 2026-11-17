import tensorflow as tf


class Box:
    def __init__(self, shape):
        self.shape = shape


def doubled(x):
    # Returns a tensor.
    return x * 2


def dynamic_shape(x):
    # `tf.shape` returns a tensor eagerly too.
    return tf.shape(x * 2)


def flattened(x):
    # Reads the static shape but returns a tensor.
    return tf.reshape(x, [x.shape[0], -1])


def shape_sizes(x):
    # Returns the shape's sizes as a list of Python integers, which is no `TensorShape`.
    return (x * 2).shape.as_list()


def box_shape(x):
    # The attribute is named `shape`, but it holds a tensor.
    return Box(x * 2).shape


x = tf.ones([4, 3])

print(doubled(x))
print(dynamic_shape(x))
print(flattened(x))
print(shape_sizes(x))
print(box_shape(x))
