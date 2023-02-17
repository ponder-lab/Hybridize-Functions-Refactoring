#!/usr/bin/python3
import tensorflow as tf

@tf.function
def func(x: tf.Tensor):
    pass

if __name__ == '__main__':
    func(5)
