#!/usr/bin/python3

import tensorflow as tf


@tf.function(autograph=True)
def func():
    pass


if __name__ == '__main__':
    func()
