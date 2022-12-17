import tensorflow as tf

@tf.function(reduce_retracing=True)
def test(x):
    return x

if __name__ == '__main__':
  number = tf.constant([1.0, 1.0])
  test(number)
