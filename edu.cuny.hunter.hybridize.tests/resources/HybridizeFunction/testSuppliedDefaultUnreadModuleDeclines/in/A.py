# Fixture for #1323: the call passing the default is in another module than the function, one
# the refactoring does not read when only B's functions are refactored, so it cannot be edited.
import tensorflow as tf

import B


def main():
    return B.f(tf.constant([1.0, 2.0]), axis=-1)


main()
