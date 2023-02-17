#!/usr/bin/python3

import tensorflow as tf


@tf.function(experimental_follow_type_hints=True)
def func():
  pass

   
if __name__ == '__main__':
    func()

