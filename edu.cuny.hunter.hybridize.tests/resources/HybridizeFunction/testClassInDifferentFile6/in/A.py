import tensorflow as tf
import B

input_data = tf.random.uniform([20, 28, 28])
model = B.SequentialModel()
result = model.__call__(input_data)
