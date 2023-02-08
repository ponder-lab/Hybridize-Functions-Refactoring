import tensorflow as tf


@tf.function(input_signature=(tf.TensorSpec([]), tf.TensorSpec([])))
def func(tensor, integer):
    return tensor + integer


if __name__ == '__main__':
  input = tf.constant(0.0)
  func(input, 2)
