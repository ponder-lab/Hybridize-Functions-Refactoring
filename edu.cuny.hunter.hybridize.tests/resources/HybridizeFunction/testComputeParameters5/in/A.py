import tensorflow as tf
  
@tf.function(jit_compile=True)
def func():
    print("Tracing")

if __name__ == '__main__':
    func()