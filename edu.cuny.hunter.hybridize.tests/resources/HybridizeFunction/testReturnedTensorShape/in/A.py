import sys

import tensorflow as tf


def attribute_shape(x):
    # Eagerly a `TensorShape`. Traced, the shape is fully defined, so the caller gets a tensor.
    return (x * 2).shape


def parameter_shape(x):
    # The shape of the parameter itself, with no tensor computation.
    return x.shape


def where_shape(x):
    # Traced, the number of rows `tf.where` finds is unknown, so tracing raises a `TypeError`.
    return tf.where(x > 0).shape


def constructed_shape(x):
    # The shape has an unknown extent, so tracing raises a `TypeError`.
    return tf.TensorShape([None, x.shape[1]])


def constructed_defined_shape(x):
    # Fully defined, so the caller gets a tensor.
    return tf.TensorShape([2, x.shape[1]])


def shape_pair(x):
    # A shape inside a returned tuple.
    y = x * 2
    return y, y.shape


def shape_dict(x):
    # A shape inside a returned dictionary.
    y = x * 2
    return {"value": y, "shape": y.shape}


def shape_list(x):
    # A shape inside a returned list. Its extent is unknown, so tracing raises a `TypeError`.
    y = x * 2
    return [y, tf.TensorShape([None, 3])]


def either_shape(x, flat):
    # One path returns a tensor and the other a shape.
    if flat:
        return tf.reshape(x, [-1])
    return x.shape


def shape_of(t):
    # A helper, which returns a shape itself.
    return t.shape


def helper_shape(x):
    # The shape comes from a helper.
    return shape_of(x * 2)


def shape_class(x):
    # The class, not a shape, beside a tensor. Traced, it raises the same `TypeError`.
    return x * 2, tf.TensorShape


x = tf.ones([4, 3])

print(attribute_shape(x).as_list())
print(parameter_shape(x).as_list())
print(where_shape(x).as_list())
print(constructed_shape(x).as_list())
print(constructed_defined_shape(x).as_list())
print(shape_pair(x)[1].as_list())
print(shape_dict(x)["shape"].as_list())
print(shape_list(x)[1].as_list())
print(either_shape(x, len(sys.argv) > 1))
print(helper_shape(x).as_list())
print(shape_class(x)[1])
