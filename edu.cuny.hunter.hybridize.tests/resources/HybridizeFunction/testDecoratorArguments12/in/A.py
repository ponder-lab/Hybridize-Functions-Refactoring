import tensorflow as tf


@tf.function(reduce_retracing=False)
def func():
  pass

 
if __name__ == '__main__':
    func()   
