import tensorflow as tf

var = (2, 2)


@tf.function(input_signature=(tf.TensorSpec(shape=var, dtype=tf.float32),))
def func(x):
  return x


if __name__ == '__main__':
    number = tf.constant(([1.0, 1.0], [2.0, 2.0]))
    func(number)
