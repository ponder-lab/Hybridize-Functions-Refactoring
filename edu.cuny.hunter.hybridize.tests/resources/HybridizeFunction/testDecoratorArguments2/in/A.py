import tensorflow as tf


@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.EQUALITY_OPERATORS)
def func(i):
  if i == 0:
    tf.print('i is zero')

 
if __name__ == '__main__':
    x = tf.constant(1)
    func(x)

