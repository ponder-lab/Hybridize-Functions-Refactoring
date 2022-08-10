import tensorflow as tf
from tensorflow.python.data.ops.structured_function import autograph

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),), autograph= False)
def func(x):
  print('Tracing with', x)
  return x

@tf.function(experimental_follow_type_hints=True)
def func2(x: tf.Tensor):
  print('Tracing')
  return x