from tf import function # tf is not TensorFlow. That's import tensorflow as tf.

@function # This is not the "function" from TensorFlow.
def func1():
    pass

if __name__ == '__main__':
    func1()
