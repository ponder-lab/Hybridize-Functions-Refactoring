import tensorflow as tf


@tf.function(experimental_autograph_options="tf.autograph.experimental.Feature.ALL")
def func():
  pass

 
if __name__ == '__main__':
    func()   
