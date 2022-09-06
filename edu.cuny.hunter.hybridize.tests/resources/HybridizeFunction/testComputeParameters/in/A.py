import tensorflow as tf

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),), autograph= False)
def func(x):
  print('Tracing with', x)
  return x

@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.EQUALITY_OPERATORS)
def func1(i):
  if i == 0:  # EQUALITY_OPERATORS allows the use of == here.
    tf.print('i is zero')

@tf.function(experimental_follow_type_hints=True)
def func2(x: tf.Tensor):
  print('Tracing')
  return x

@tf.function(experimental_implements="google.matmul_low_rank_matrix")
def func3():
    pass
  
@tf.function(jit_compile=True)
def func4():
    print("Tracing")

@tf.function(reduce_retracing=True)
def func5(x, y):
  return x ** 2 + y
  
@tf.function(experimental_compile=True)
def func6():
     print("Tracing")
 
if __name__ == '__main__':
    number = tf.constant([1.0, 1.0])
    x = tf.constant(1)
    y = tf.constant(2)
    func(number)
    func1(x)
    func2(x)
    func3()
    func4()
    func5(x, y)
    func6()
    