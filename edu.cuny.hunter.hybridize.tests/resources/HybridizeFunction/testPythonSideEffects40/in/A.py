# From  https://www.tensorflow.org/guide/function#using_python_iterators_and_generators

import tensorflow as tf


@tf.function
def buggy_consume_next(iterator):
    tf.print("Value:", next(iterator))


iterator = iter([1, 2, 3])

buggy_consume_next(iterator)
# This reuses the first value from the iterator, rather than consuming the next value.
buggy_consume_next(iterator)
buggy_consume_next(iterator)
