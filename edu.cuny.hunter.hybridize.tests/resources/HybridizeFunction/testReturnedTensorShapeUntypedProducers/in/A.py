import tensorflow as tf


def method_shape(x):
    # The shape `x.shape` reads, read by `get_shape`. Traced, the caller gets a tensor.
    return (x * 2).get_shape()


def dataset_shapes(dataset):
    # Eagerly a structure of `TensorShape`s. Traced, a shape with an unknown extent raises a
    # `TypeError`, and a fully defined one comes back as a tensor.
    return tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)


def spec_shape(dataset):
    # The shape of the dataset's only component, read off its spec.
    return dataset.element_spec.shape


def legacy_shapes(dataset):
    # The same shape, by the TensorFlow 1 spelling.
    return tf.compat.v1.data.get_output_shapes(dataset)


x = tf.ones([4, 3])
unknown = tf.data.Dataset.from_tensor_slices(x).batch(2)
known = tf.data.Dataset.from_tensor_slices(x).batch(2, drop_remainder=True)

print(method_shape(x).as_list())
print(dataset_shapes(unknown).as_list())
print(dataset_shapes(known).as_list())
print(spec_shape(unknown).as_list())
print(spec_shape(known).as_list())
print(legacy_shapes(unknown).as_list())
print(legacy_shapes(known).as_list())
