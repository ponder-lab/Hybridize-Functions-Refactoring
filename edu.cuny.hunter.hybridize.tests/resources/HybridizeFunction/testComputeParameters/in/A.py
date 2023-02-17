#!/usr/bin/python3

import tensorflow as tf


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
def func(x):
  return x

if __name__ == '__main__':
    number = tf.constant([1.0, 1.0])
    func(number)
