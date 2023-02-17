#!/usr/bin/python3

import tensorflow as tf


@tf.function(reduce_retracing=True)
def func():
  pass

 
if __name__ == '__main__':
    func()   
