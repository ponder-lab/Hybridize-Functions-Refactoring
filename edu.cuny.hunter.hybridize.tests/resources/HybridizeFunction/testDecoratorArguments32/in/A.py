import tensorflow as tf

var = False

@tf.function(reduce_retracing=var)
def func(x):
  return x


if __name__ == '__main__':
    func(tf.constant(1))
