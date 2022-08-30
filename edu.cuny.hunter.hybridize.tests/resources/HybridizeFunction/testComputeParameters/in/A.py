import tensorflow as tf

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),), autograph= False)
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
def func3(x, y):
    pass
  
@tf.function(jit_compile=True)
def func4(images, labels):
    images, labels = cast(images, labels)

    with tf.GradientTape() as tape:
      predicted_labels = layer(images)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=predicted_labels, labels=labels
      ))
    layer_variables = layer.trainable_variables
    grads = tape.gradient(loss, layer_variables)
    optimizer.apply_gradients(zip(grads, layer_variables))

@tf.function(reduce_retracing=True)
def func5():
  return x ** 2 + y
  
@tf.function(experimental_compile=True)
def func6(self, I):
     shape = tf.shape(I)
     y, x = tf.range(0, shape[1]), tf.range(0, shape[2])
     grid = tf.meshgrid(x, y)
     
     return grid
 
if __name__ == '__main__':
    func()
    func1()
    func2()
    func3()
    func4()
    func5()
    func6()