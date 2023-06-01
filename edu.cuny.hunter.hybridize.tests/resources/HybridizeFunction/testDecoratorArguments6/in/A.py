import tensorflow as tf


@tf.function(autograph=False)
def func():
  pass

 
if __name__ == '__main__':
    func()   
