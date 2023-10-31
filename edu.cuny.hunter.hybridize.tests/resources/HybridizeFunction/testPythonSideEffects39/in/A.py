# From https://www.tensorflow.org/guide/function#changing_python_global_and_free_variables.
import tensorflow as tf


class Model(tf.Module):

  def __init__(self):
    self.v = tf.Variable(0)
    self.counter = 0

  @tf.function
  def __call__(self):
    if self.counter == 0:
      # A python side-effect
      self.counter += 1
      self.v.assign_add(1)

    return self.v


m = Model()
for n in range(3):
  print(m.__call__().numpy())  # prints 1, 2, 3
