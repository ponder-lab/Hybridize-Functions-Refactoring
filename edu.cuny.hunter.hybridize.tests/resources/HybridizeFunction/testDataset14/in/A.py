# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/TextLineDataset.

import tensorflow as tf


def func(a):
    assert isinstance(a, tf.Tensor)


with open("/tmp/text_lines0.txt", "w") as f:
    f.write("the cow\n")
    f.write("jumped over\n")
    f.write("the moon\n")

with open("/tmp/text_lines1.txt", "w") as f:
    f.write("jack and jill\n")
    f.write("went up\n")
    f.write("the hill\n")

dataset = tf.data.TextLineDataset(["/tmp/text_lines0.txt", "/tmp/text_lines1.txt"])

for element in dataset:
    func(element)
