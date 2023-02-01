import tensorflow as tf


@tf.function(experimental_autograph_options=None)
def func():
  pass

 
if __name__ == '__main__':
    func()   
