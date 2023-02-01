import tensorflow as tf


@tf.function(input_signature=None)
def func(x):
  return x

if __name__ == '__main__':
    number = tf.constant([1.0, 1.0])
    func(number)
