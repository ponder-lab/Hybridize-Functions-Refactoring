import tensorflow as tf

@tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
def func(x):
  return x

if __name__ == '__main__':
  func(tf.constant(2.))
