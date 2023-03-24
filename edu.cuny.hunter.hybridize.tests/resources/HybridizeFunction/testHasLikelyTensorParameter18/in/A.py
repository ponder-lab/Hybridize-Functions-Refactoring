import tensorflow as tf

# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#features.
# func's closure may include tf.Tensor and tf.Variable objects:


# @tf.function
def f():
  return x ** 2 + y


x = tf.constant([-2, -3])
y = tf.Variable([3, -2])
r = f()
print(r)
