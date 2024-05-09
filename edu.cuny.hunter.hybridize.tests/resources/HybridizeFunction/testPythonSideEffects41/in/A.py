# From  https://www.tensorflow.org/guide/function#using_python_iterators_and_generators

import tensorflow as tf


@tf.function
def good_consume_next(iterator):
    # This is ok, iterator is a tf.data.Iterator
    tf.print("Value:", next(iterator))


ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
iterator = iter(ds)
good_consume_next(iterator)
good_consume_next(iterator)
good_consume_next(iterator)
