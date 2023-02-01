import tensorflow as tf

@tf.function(input_signature=[])
def func(x):
  return x

if __name__ == '__main__':
  func(tf.constant(2.))
