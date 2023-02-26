import tensorflow as tf


@tf.function(autograph=False)
@tf.function(jit_compile=True)
def func(x):
  return x

  
if __name__ == '__main__':
    func(tf.constant(1))

