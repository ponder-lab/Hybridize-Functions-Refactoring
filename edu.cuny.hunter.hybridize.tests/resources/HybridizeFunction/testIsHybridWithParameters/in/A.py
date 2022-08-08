import tensorflow as tf

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
  print("Tracing with", x)
  return tf.where(x % 2 == 0, x // 2, 3 * x + 1)

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def g(x):
  print('Tracing with', x)
  return x

@tf.function(experimental_follow_type_hints=True)
def f_with_hints(x: tf.Tensor):
  print('Tracing')
  return x
