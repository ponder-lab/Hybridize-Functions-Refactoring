import tensorflow as tf

var = (tf.TensorSpec(shape=[None], dtype=tf.float32),)


@tf.function(input_signature=var)
def func(x):
  return x


if __name__ == '__main__':
    number = tf.constant([1.0, 1.0])
    func(number)
