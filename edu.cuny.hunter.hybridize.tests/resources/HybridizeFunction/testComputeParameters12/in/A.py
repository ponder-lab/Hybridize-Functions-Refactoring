import tensorflow as tf

@tf.function(autograph=False)
@tf.function(jit_compile=True)
def func():
  pass
  
if __name__ == '__main__':
    func()

