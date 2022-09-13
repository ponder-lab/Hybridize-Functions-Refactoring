import tensorflow as tf


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),), autograph=False)
def func(x):
  print('Tracing with', x)
  return x

if __name__ == '__main__':
    number = tf.constant([1.0, 1.0])
    func(number)
    
    
