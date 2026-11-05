# Fixture for dicts of ints whose keys the analysis lists, the complete counterpart of
# testIncompleteContainerUnknown: `lookup` receives only such a dict and a string, and `scale`
# receives one beside a tensor. Neither dict holds a tensor, and the analysis can say so.
import tensorflow as tf


def lookup(table, key):
    return table[key]


def scale(x, table):
    return x * table["a"]


print(lookup({"a": 1, "b": 2}, "a"))
print(scale(tf.ones([2]), {"a": 1, "b": 2}))
