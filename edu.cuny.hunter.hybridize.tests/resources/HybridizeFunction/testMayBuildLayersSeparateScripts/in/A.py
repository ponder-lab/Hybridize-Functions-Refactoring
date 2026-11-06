import tensorflow as tf

from B import Net

model = Net()
model(tf.ones([2, 4]))
